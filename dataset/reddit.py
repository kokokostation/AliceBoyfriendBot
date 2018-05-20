import re
import os
import json


def prepare_reddit(from_tsv, to_dir, context_len):
    if os.path.exists(to_dir):
        print("{} already exists".format(to_dir))
        return

    os.makedirs(to_dir, exist_ok=True)

    with open(from_tsv, "r") as fin:
        c = 0
        dialogs_buffer = []
        while True:
            line = fin.readline()
            if not line:
                break
            c += 1
            if c % 10000 == 0:
                print(c)
            msgs = line.strip().split("\t")[2]
            assert msgs.startswith("value=")
            msgs = msgs[6:].split(":egassem:")
            for i, msg in enumerate(msgs):
                author, orig_message = msg.replace("\\t", " ").replace("\\n", " ").split(
                    ":rohtua:")

                author = author.strip()

                msgs[i] = [author, orig_message, convert(orig_message)]

            for i in range(len(msgs) - context_len):
                dialogs_buffer.append(msgs[i:i + context_len + 1])

            if c % 1000 == 0:
                with open(os.path.join(to_dir, str(c)), "w") as fout:
                    json.dump(dialogs_buffer, fout)
                dialogs_buffer = []
        with open(os.path.join(to_dir, str(c)), "w") as fout:
            json.dump(dialogs_buffer, fout)

    print("done")


def convert(message):
    message = message.lower()

    message = re.sub("[!¡]{2,}", " MXT ", message)
    message = re.sub("[?¿]{2,}", " MQT ", message)
    message = re.sub("[.]{2,}", " MDT ", message)
    message = re.sub("[,]{2,}", " MPT ", message)

    message = re.sub("[!¡]", " OXT ", message)
    message = re.sub("[?¿]", " OQT ", message)
    message = re.sub("[.]", " ODT ", message)
    message = re.sub("[,]", " OPT ", message)

    message = re.sub("[(]+[-]*[=:;x]+", " FT ", message)
    message = re.sub("[=:;x]+[-]*[)d]+", " FT ", message)

    message = re.sub("[)]+[-,'\"]*[=:;x]+", " ST ", message)
    message = re.sub("[=:;x]+[-,'\"]*[(]+", " ST ", message)

    message = re.sub("<3", " LT ", message)
    message = re.sub(":\*", " KT ", message)

    message = re.sub("\d+", " NT ", message)

    message = re.sub("[^\w ]+", " ", message)
    message = re.sub(" +", " ", message).strip()

    return message


def is_unique(entry):
    return entry[-1]['processed_body'] not in [e['processed_body'] for e in entry[:-1]]
