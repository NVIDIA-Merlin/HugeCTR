set -e

# -------- get TF version -------------- #
TfVersion=`python3 -c "import tensorflow as tf; print(tf.__version__.strip().split('.'))"`
TfMajor=`python3 -c "print($TfVersion[0])"`
TfMinor=`python3 -c "print($TfVersion[1])"`

if [ "$TfMajor" -eq 2 ]; then
    bash tf2/script.sh; exit 0;
else
    bash tf1/script.sh; exit 0;
fi