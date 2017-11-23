cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $cur_dir
model_dir=${cur_dir}/train_cache/icvl_training_s2_f128_daug_um_v1/
if ! [ -d $model_dir ]; then
    mkdir -p $model_dir
fi

cd $model_dir
url=https://polybox.ethz.ch/index.php/s/f9EWUGSpTeKmFDo/download
fname=icvl.tar.gz

if [ -f $fname ]; then
    echo "file already exists, no need to download again"
else
    echo "downloading the pretrained model(62M)..."
    wget $url
    mv download $fname
fi

echo "unzipping..."
tar xvzf $fname
mv icvl/*.* ./
rmdir icvl/

echo "done."