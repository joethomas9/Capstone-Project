# Aircraft Recognition Software

> Developer: Joseph Thomas
> Contact: jt19712@essex.ac.uk
> 
> Undergraduate student of Computer Science and Electronic Engineering at 
> the University of Essex.

### Installation of pre-requisits

The following libraries are required for execution of the program

> opencv: https://pypi.org/project/opencv-python/
> >'pip install opencv-python'
>
> numpy: https://numpy.org/install/
> >'pip install numpy'
> 
> scikit learn: https://scikit-learn.org/stable/install.html
> >'pip install scikit-learn'
> 
> 
> tensorflow: 
>*note, installation version of tensorflow will depend on machine, please see link attached*
> https://www.tensorflow.org/install/pip
> > 'pip install tensorflow-{*version compatible to your system*}'

### Program Execution

The program is executed via the terminal using command ./main.py {training mode (True/False)}. 
If you wish to use the program for creating a training set, use True, else False.
The program comes with a default training set, however you are welcome to replace 
this with a training set of your own. 

####Training
If you wish to use the program to simply extract any contours from the image, upon execution
write True following the command to execute the file on the command line as shown above.
Please note, in training mode all extractions that are generated are stored within 
the same file in ../main/training/extracts_training, it is up to yourself to categorise them. 

####Classifying
When looking to run the model to classify aircraft from your own test images, place the images
within main/data/images, and the program will classify their content. Function behaviour
is discussed in comments across the program, if you are not sure what a function does, or
are looking to alter the program please use those as guidance. If you have any other questions,
please contact using the information listed above.




