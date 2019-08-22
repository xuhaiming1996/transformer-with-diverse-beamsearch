#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=32000



if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=src
tgt=tgt
prep=bdzd.tokenized.src-tgt
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep






#echo "pre-processing train data..."
#for l in $src $tgt; do
#    f=train.sens.$l
#    tok=train.sens.tok.$l
#    cat $orig/$f | \
#    grep -v '<url>' | \
#    grep -v '<talkid>' | \
#    grep -v '<keywords>' | \
#    sed -e 's/<title>//g' | \
#    sed -e 's/<\/title>//g' | \
#    sed -e 's/<description>//g' | \
#    sed -e 's/<\/description>//g' | \
#    perl $TOKENIZER -threads 8 -l zh > $tmp/$tok
#    echo ""
#done


#
#perl $CLEAN -ratio 1.5 $tmp/train.sens.tok $src $tgt $tmp/train.sens.clean 1 175
#for l in $src $tgt; do
#    perl $LC < $tmp/train.sens.clean.$l > $tmp/train.sens.$l
#done
#
#echo "pre-processing test data..."
#for l in $src $tgt; do
#    o=$orig/test.sens.$l
#    f=$tmp/test.sens.$l
#    echo $o $f
#
#    cat $o | \
#    grep -v '<url>' | \
#    grep -v '<talkid>' | \
#    grep -v '<keywords>' | \
#    sed -e 's/<title>//g' | \
#    sed -e 's/<\/title>//g' | \
#    sed -e 's/<description>//g' | \
#    sed -e 's/<\/description>//g' | \
#    perl $TOKENIZER -threads 8 -l zh  | \
#    perl $LC > $f
#    echo ""
#
#done
#
#
#echo "creating train, test..."
#for l in $src $tgt; do
#    cat $tmp/train.sens.$l > $tmp/train.$l
#    cat $tmp/test.sens.$l > $tmp/test.$l
#done
#
TRAIN=$tmp/train.src-tgt
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
