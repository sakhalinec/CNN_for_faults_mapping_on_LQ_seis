## Convolutional neural network(CNN) for tectonic faults mapping on low-quality seismic data
- This CNN was developed specifically for Low-quality seismic data which is common for survey conducted in onshore setting resulting in noisy data. Good interpretation results are achieved through pre-processing of input seismic data and training CNN to identify faults on additionally processed seismic data
- Though originally developed for low quality cases it will provide good results on High and Moderate - quality seismic data.
- CNN is based on U-net architecture
- It is originally developed for processing using Google Colaboratory
- SEG-Y file given in this project can be process using free version of Google  Colaboratory, bigger SEG-Y files will require subscription to get more computing capacity
- Running on desktop will require minor code adjustment
- To try it without adjusting code, create a folder: "NN_for_tectonic_faults_mapping" on Google Drive and upload repo into it

## Example of Low-quality seismic data processing

### Original seismic 
Seismic survey conducted in onshore environment in western Siberia
![3D Input seismic volume](https://user-images.githubusercontent.com/112522254/229543197-cb8bacce-59a6-4559-999a-a2a913135471.jpg)

### Seismic with interpreted Faults
![3D volume with interpreted tectonic faults](https://user-images.githubusercontent.com/112522254/229543328-c5ce7952-9d8d-4546-9de1-d8d5211c92d5.jpg)


## License
This CNN is released under a creative commons license which allows for personal and research use only. 
You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
