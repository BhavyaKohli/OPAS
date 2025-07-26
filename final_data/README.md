# Datasets

Download `final_data.zip` from [here](https://drive.google.com/drive/folders/1EVI_t2oObqU0uCeHaX1NrBUvg9Uvn3_H?usp=sharing) and unzip its contents into this directory to have three folders (audio (==the music dataset), speech, cifar) each with a single `.hdf5` file inside, along with this README in the current folder. 

After extraction, run `ln -s cifar cifar-large` to create a symbolic link which points to `cifar/`, for compativility with some assets which may try to access `cifar-large/`