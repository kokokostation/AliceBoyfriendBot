#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from datetime import datetime
from collections import defaultdict

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


class TelegramBot:
    DUMP_PREFIX = "TgHistory"

    def __init__(self, tg_token, dump_dir, applier=None, context_length=3):
        self.__bot = Updater(tg_token)
        self.__dump_dir = dump_dir
        self.__history = defaultdict(lambda: ["Mama", "Here I am"])
        self.context_length = context_length
        self.__applier = applier

    # Model specific func tools

    def set_applier(self, applier):
        self.__applier = applier
        return self

    def wrap_with_context(self, author, message):
        context = self.__history[author] + [message]
        context = context[-self.context_length:]
        self.__history[author] = context
        return context

    def generate_reply(self, author, message):
        w_message = self.wrap_with_context(author, message)
        reply = self.__applier.reply(w_message)
        self.wrap_with_context(author, reply)
        return reply

    def try_dump_history(self):
        if not self.__history:
            logging.warning("No Telegram Bot history initialized to dump")
        cur_time = datetime.now().strftime("%Y-%m-%d[%H:%M:%S]")
        # cur_time = "last"
        filename = TelegramBot.DUMP_PREFIX + "_" + cur_time + ".json"
        filepath = os.path.join(self.__dump_dir, filename)
        with open(filepath, "w") as file:
            json.dump(self.__history, file)

    def try_load_history(self):
        filenames = [
            fn for fn in os.listdir(self.__dump_dir)
            if fn.startswith(TelegramBot.DUMP_PREFIX)
        ]
        if not filenames:
            logging.warning("No Telegram Bot history found in {}".format(self.__dump_dir))
            return
        filename = max(filenames)
        filepath = os.path.join(self.__dump_dir, filename)
        with open(filepath, "r") as file:
            dumped_history = json.load(file)
        assert isinstance(self.__history, dict), "history must be a dict(chat_id->list)"
        self.__history.update(dumped_history)

    # Bot functions

    def initialize(self, bot, update):
        chat_id = update.message.chat.id
        reply = "Hi! I'm Alice Boyfriend. Do you have problems with that?"

        self.__history[chat_id].append(reply)
        update.message.reply_text(reply)

    def message_handler(self, bot, update):
        chat_id = update.message.chat.id
        message = update.message.text

        reply = self.generate_reply(str(chat_id), message)

        update.message.reply_text(reply)

    def drop_context(self, bot, update):
        chat_id = update.message.chat.id

        del self.__history[str(chat_id)]

        reply = "your context dropped"

        update.message.reply_text(reply)

    def start(self):
        """Start the bot."""
        assert self.__applier, "load_model should be called before starting the bot"
        self.try_load_history()

        dp = self.__bot.dispatcher
        dp.add_handler(CommandHandler("start", self.initialize))
        dp.add_handler(CommandHandler("drop_context", self.drop_context))
        dp.add_handler(MessageHandler(Filters.text, self.message_handler))

        self.__bot.start_polling()
        self.__bot.idle()
