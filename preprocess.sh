NAME=iwslt14.tokenized.de-en
TEXT=examples/translation/$NAME
python3 preprocess.py \
    --source-lang de --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir data-bin-joined-dict/$NAME \
    --joined-dictionary \
    --workers 8
