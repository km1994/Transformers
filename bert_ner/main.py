from net.bert_ner import Bert_CRF
from data_loader import create_batch_iter
from train import fit
from util.porgress_util import ProgressBar
from Config import Config
args = Config()
from util.Logginger import init_logger
logger = init_logger("bert_ner", logging_path=args.log_path)
def start():
    model = Bert_CRF()
    print('create_iter')
    train_iter, num_train_steps = create_batch_iter("train")
    eval_iter = create_batch_iter("valid")
    print('create_iter finished')

    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    logger.info('fit')

    fit(
        model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1
    )


if __name__ == '__main__':
    start()
