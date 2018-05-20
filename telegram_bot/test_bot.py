import os
import sys

# codebase_path = os.path.join(os.getcwd(), "..")
codebase_path = os.getcwd()
print("Adding {} to PATH".format(codebase_path))
sys.path.append(codebase_path)

from telegram_bot import TelegramBot
import logging
import argparse


parser = argparse.ArgumentParser(description='Run telegram bot for alice boyfriend.')
parser.add_argument(
    '--token',
    type=str,
    help='Telegram bot token',
    dest='tg_bot_token',
)
parser.add_argument(
    '--model_dir',
    type=str,
    required=False,
    help='Model dir where all params are stored',
    dest='model_dir',
    default="/data/reddit/models/x_prod/",
)
parser.add_argument(
    '--model_id',
    type=str,
    required=False,
    help='Model name (baseline, lstm or ranking)',
    dest='model_id',
    default="baseline",
)

parser.add_argument(
    '--gpu_device_num',
    type=str,
    required=False,
    help='Value to be stored in CUDA_VISIBLE_DEVICES',
    dest='gpu_device_num',
    default="7",
)


def load_model(model_id):
    if model_id == 'baseline':
        from application.applier_constructors import make_baseline as make_model
    elif model_id == 'lstm':
        from application.applier_constructors import make_lstm as make_model
    elif model_id == 'ranking':
        from application.applier_constructors import make_ranking as make_model
    else:
        raise NotImplementedError

    ap = make_model()
    return ap


def main(tg_bot_token, model_dir, model_id):
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading the model...")
    applier = load_model(model_id)

    logging.info(" Building bot...")
    bot = TelegramBot(
        tg_token=tg_bot_token,
        dump_dir=model_dir,
    ).set_applier(
        applier=applier,
    )
    try:
        logging.info("Starting the bot...")
        bot.start()
    finally:
        bot.try_dump_history()


if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device_num

    main(args.tg_bot_token, args.model_dir, args.model_id)
