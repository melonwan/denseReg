cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $cur_dir

cache_dir=${cur_dir}/msra_model
if ! [ -d $cache_dir ]; then
    mkdir $cache_dir
fi
cd $cache_dir

fname=msra.tar.gz
url=https://polybox.ethz.ch/index.php/s/B2W1ngyUAitsv2e/download
if [ -f $fname ]; then
    echo "file already exists, no need to download again"
else 
    echo "downloading the pretrained model(566M)..."
    wget $url
    mv download $fname
fi
echo "unzipping..."
tar xvzf $fname


cd $cur_dir
for pid in {0..8}; do
    tar_dir=${cur_dir}/train_cache/msra_P${pid}_training_s2_f128_daug_um_v1/
    src_dir=${cache_dir}/msra/P${pid}/
    mv $src_dir $tar_dir
done

rmdir ${cache_dir}/msra
echo "done."