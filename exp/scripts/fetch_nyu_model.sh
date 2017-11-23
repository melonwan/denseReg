cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $cur_dir
model_dir=${cur_dir}/train_cache/nyu_training_s2_f128_daug_um_v1/
if ! [ -d $model_dir ]; then
    mkdir -p $model_dir
fi

cd $model_dir
url=https://polybox.ethz.ch/index.php/s/Q4GS7bgRRM3zK5J/download
fname=nyu.tar.gz

if [ -f $fname ]; then
    echo "file already exists, no need to download again"
else
    echo "downloading the pretrained model(61M)..."
    wget $url
    mv download $fname
fi

echo "unzipping..."
tar xvzf $fname
mv nyu/*.* ./
rmdir nyu/

echo "done."