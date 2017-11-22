this directory holds the benchmark dataset.
## download the dataset
Please download and unpack the corresponding dataset, and create softlink of the corresponding dataset here.
```bash
ln -s /path/to/nyu ./nyu
ln -s /path/to/icvl ./icvl
ln -s /path/to/msra ./msra
```
- could go to the [link](https://github.com/moberweger/deep-prior-pp) to get the link to download the corresponding dataset(thanks @markus for providing the links and thanks @liuhao for providing the msra dataset download)

## prepare the TFRecord files
We convert all the source datas to TFRecord files to accelarate data loading speed.
For a certain dataset(icvl, nyu, msra), run
```bash
python data/${dataset}.py
```
to create the corresponding tf file records.
