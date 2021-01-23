
### To access stimulus file to change trial structure
at Home/data/2020_test/nagel_stimulus_parameters.yaml


#### Running an experiment

Open a terminal (`Ctrl-Alt-t` is the shortcut for this)

```
cd ~/data/2020_test
roslaunch nagel_laminar nagel_laminar.launch
```

A video will pop up for you to specify the tracking ROIs (window titled `Manual
ROI selection`). Click each corner of the ROIs you want to track, and press any
key to finish entering a single ROI. Once you are done entering all ROIs, press
the `Esc` key to save all the ROI definitions and start the tracking.

Once terminal messages stop being printed regarding the tracking starting,
switch back to the terminal you used to start the tracking, and press Enter to
start the stimulus program.


### To convert video file 
Converts .bag to .avi file for fast viewing.
Switch to directory of current experiment.

```
rosrun multi_tracker bag2vid.py
```

Will output:
```
Note: use this command to make a mac / quicktime friendly video: avconv -i test.avi -c:v libx264 -c:a copy outputfile.mp4
```

Note: terminal shortcut for opening file is: `xdg-open <name-of-file>` (equivalent to clicking on the file in the file explorer).


### PID
The same command to run a full experiment above can be used for PID, if you add additional options as follows:
```
roslaunch nagel_laminar nagel_laminar.launch pid:=True stimuli_only:=True
```

If you'd like to PID while doing video data acquisition / tracking, you may omit the `stimuli_only:=True` flag.


### Setting up the Arduino
Stimulus delivery with this system requires specific code to be uploaded to the Arduino. This code should typically just stay on the Arduino, and thus you should not need to re-upload it each day. However, if you accidentally uploaded some other code to the Arduino, or are using a new Arduino, you can upload the code as follows:
```
cd arduino-1.8.3/
./arduino # will open the Arduino IDE
```

In the Arduino IDE:
- Open `File`->`Examples`->`stimuli`->`stimuli`
- Press upload (OK to ignore the compiler warnings)

