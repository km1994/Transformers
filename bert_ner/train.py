import time
import torch
import numpy as np
from pytorch_pretrained_bert.optimization import BertAdam
from util.plot_util import loss_acc_plot

from util.model_util import save_model, load_model
import warnings
from util.score import score_predict
from util.metrics import gen_metrics,mean
from Config import Config
from util.metrics import get_chunk
from util.score import get_tags
from sklearn.metrics import f1_score,recall_score,precision_score
args = Config()
from util.Logginger import init_logger
logger = init_logger("bert_ner", logging_path=args.log_path)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
warnings.filterwarnings('ignore')

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def fit(model, training_iter, eval_iter, num_epoch, pbar, num_train_steps, verbose=1):
    # ------------------判断CUDA模式----------------------
    device = torch.device(args.device)

    # ---------------------优化器-------------------------
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in args.no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in args.no_decay)], 'weight_decay': 0.0}]

    t_total = num_train_steps

    ## ---------------------GPU半精度fp16-----------------------------
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    ## ------------------------GPU单精度fp32---------------------------
    else:
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            warmup=args.warmup_proportion,
            t_total=t_total
        )
    # ---------------------模型初始化----------------------
    if args.fp16:
        model.half()

    model.to(device)

    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []

    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy
    }

    # ------------------------训练------------------------------
    best_f1 = 0
    start = time.time()
    global_step = 0
    logger.info(f"start train!!!")
    for e in range(num_epoch):
        model.train()
        train_accs = []
        for step, batch in enumerate(training_iter):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask = batch
            bert_encode = model(
                input_ids, segment_ids, input_mask
            ).cpu()
            train_loss = model.loss_fn(
                bert_encode=bert_encode, tags=label_ids, output_mask=output_mask
            )

            if args.gradient_accumulation_steps > 1:
                train_loss = train_loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(train_loss)
            else:
                train_loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            predicts = model.predict(bert_encode, output_mask)
            label_ids = label_ids.view(1, -1).squeeze().cpu()
            predicts = predicts.view(1, -1).squeeze().cpu()
            label_ids = label_ids[label_ids != -1]
            predicts = predicts[predicts != -1]
            train_acc = cul_acc(predicts, label_ids)
            train_accs.append(train_acc)
            logger.info(f"step:{step}=>loss:{train_loss.item()},train_acc:{train_acc}")

        # -----------------------验证----------------------------
        model.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_acc = 0
        eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = 0, 0, 0, 0, 0
        with torch.no_grad():
            eval_precisions,eval_recalls,eval_f1s = [],[],[]
            for step, batch in enumerate(eval_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask = batch
                bert_encode = model(input_ids, segment_ids, input_mask).cpu()
                eval_los = model.loss_fn(
                    bert_encode=bert_encode, tags=label_ids, output_mask=output_mask
                )
                eval_loss = eval_los + eval_loss
                count += 1
                predicts = model.predict(bert_encode, output_mask)

                label_ids = label_ids.view(1, -1).squeeze()
                predicts = predicts.view(1, -1).squeeze()
                label_ids = label_ids[label_ids != -1]
                predicts = predicts[predicts != -1]
                label_ids = label_ids.cpu().numpy().tolist()
                predicts = predicts.cpu().numpy().tolist()

                precision = precision_score(predicts,label_ids, average='weighted')
                recall = recall_score(predicts,label_ids, average='weighted')
                f1 = f1_score(predicts,label_ids, average='weighted')

                eval_recalls.append(recall)
                eval_precisions.append(precision)
                eval_f1s.append(f1)

            eval_precision = mean(eval_precisions)
            eval_recall = mean(eval_recalls)
            eval_f1 = mean(eval_f1s)

            # 保存最好的模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                save_model(model, args.output_dir)

            info = '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - p:%4f - r:%4f - f1:%4f - best f1:%4f\n' % (e + 1,
                   train_loss.item(),
                   eval_loss.item() / count,
                   mean(train_accs),
                   eval_acc,
                   eval_precision, eval_recall, 
                   eval_f1, best_f1)
            
            logger.info(info)

            if e % verbose == 0:
                train_losses.append(train_loss.item())
                train_accuracy.append(mean(train_accs))
                eval_losses.append(eval_loss.item() / count)
                eval_accuracy.append(eval_acc)

    loss_acc_plot(history)

# 功能：计算 acc
def cul_acc(y_true,y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    correct = np.sum((y_true == y_pred).astype(int))
    acc = correct / y_pred.shape[0]
    return acc
