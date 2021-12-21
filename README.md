
### Configuration
Stimulus parameters can be found in `~/data/2020_test/nagel_stimulus_parameters.yaml`

Other required configuration files are:
- `data_association_parameters.yaml`
- `debug_flags.yaml`
- `delta_video_parameters.yaml`
- `liveviewer_parameters.yaml`
- `pointgrey_blackfly.yaml`
- `roi_finder.yaml`
- `tracker_parameters.yaml`
- `kalman_parameters.py`
- `camera_info/0.yaml`

You will likely not need to change any of these files apart from
`nagel_stimulus_parameters.yaml`, though if you'd like to run experiments from
another directory you must copy all of these files to that new directory.

### Running an experiment

Open a terminal (`Ctrl-Alt-t` or clicking on its icon), and enter the following
commands:
```
cd ~/data/2020_test
roslaunch nagel_laminar nagel_laminar.launch
```

A video will pop up for you to specify the tracking ROIs (window titled `Manual
ROI selection`). Click each corner of the ROIs you want to track, and press any
key (e.g. spacebar, but anything besides keys mentioned in "Other commands..."
section below) to finish entering a single ROI. Once you are done entering all
ROIs, press the `Esc` key to save all the ROI definitions and start the
tracking.

Once terminal messages stop being printed regarding the tracking starting,
switch back to the terminal you used to start the tracking, and press Enter to
start the stimulus program.

##### Other commands in ROI entry window
- `c`: clears ROIs and deletes cached ROIs
- `x`: clears the points that have been entered but not yet turned into an ROI
- `z`/`u`: undo
- `y`/`r`: redo

Other reserved keys are: `l`, `d`, `s`

##### Stopping early

You may abort an experiment by pressing `Ctrl-C` inside the terminal you ran
`roslaunch` from. Press it once and wait for everything to shutdown. There may
be a few odor pulses delivered after shutdown, though this will not interfere
with another stimulus program from being started correctly after the shutdown.

#### Inputs
See the "Configuration" section above.

#### Outputs
A folder will be created inside the directory where you ran 
- `choice_<YYYYMMDD>_<HHMMSS>_delta_video.bag`
- `choice_<YYYYMMDD>_<HHMMSS>_deltavideo_bgimg_*_N<ROI>.png`
- `choice_<YYYYMMDD>_<HHMMSS>_N<ROI>_trackedobjects.hdf5`
- `choice_<YYYYMMDD>_<HHMMSS>_stimuli.p`
- `choice_<YYYYMMDD>_<HHMMSS>_..._parameters.yaml`
- `compressor_rois_<YYYYMDD>_<HHMSS>_N1.yaml`
- `full_background.png`


### Testing the valves
```
cd ~/data/2020_test
roslaunch nagel_laminar test_valves.launch
```

This will cycle through the valves from lowest pin number to highest. Listen for
the valve clicks. Press `Ctrl-C` to quit.

You should do this at least at the beginning of every experimental day, and
probably also at the end.


### Convert `.bag` to `.avi` file

To convert the `.bag` file to a format you can open in a video player:
```
# Change directory to inside the output of the most recent experiment, e.g.
cd ~/data/2020_test/choice_<YYYYMMDD>_<HHMMSS>

rosrun nagel_laminar bag2vid
```

The .avi file will also have an overlay in the bottom middle of the screen that
says "Odor On" on the frames when the odor is being delivered.

Note: to open video `vlc <name-of-file>`. You can change the speed of the playback
with the `[` and `]` keys.

#### Inputs
- `choice_<YYYYMMDD>_<HHMMSS>_delta_video.bag`
- `choice_<YYYYMMDD>_<HHMMSS>_deltavideo_bgimg_*_N<ROI>.png`
- `choice_<YYYYMMDD>_<HHMMSS>_stimuli.p`

#### Outputs
- `choice_<YYYYMMDD>_<HHMMSS>_delta_video.avi`
- `frametimes.txt`
- `choice_<YYYYMMDD>_<HHMMSS>_stimuli.csv`


### Convert `.hdf5` tracking data to `.csv`

```
rosrun multi_tracker hdf5_to_csv.py
```

#### Inputs
- `choice_<YYYYMMDD>_<HHMMSS>_N<ROI>_trackedobjects.hdf5`

#### Outputs
- `choice_<YYYYMMDD>_<HHMMSS>_N<ROI>_trackedobjects.csv`


### PID
The same command to run a full experiment above can be used for PID, if you add
additional options as follows:
```
roslaunch nagel_laminar nagel_laminar.launch pid:=True stimuli_only:=True
```

If you'd like to PID while doing video data acquisition / tracking, you may omit
the `stimuli_only:=True` flag.

#### Inputs
- `nagel_stimulus_parameters.yaml`
- `labjack.yaml`

#### Outputs
A directory will be created named as `<YYYYMMDD>_<HHMMSS>` (note lack of the
hostname prefix in the directories created when also tracking), containing:
- `choice_<YYYYMMDD>_<HHMMSS>_stimuli.p`
- `pid.csv`


### Setting up the Arduino
Stimulus delivery with this system requires specific code to be uploaded to the
Arduino. This code should typically just stay on the Arduino, and thus you
should not need to re-upload it each day. However, if you accidentally uploaded
some other code to the Arduino, or are using a new Arduino, you can upload the
code as follows:
```
cd arduino-1.8.3/
./arduino # will open the Arduino IDE
```

In the Arduino IDE:
- Open `File`->`Examples`->`stimuli`->`stimuli`
- Press upload (OK to ignore the compiler warnings)

If you don't see an entry for `stimuli` under the example menu, see the
instructions for how to set up the stimuli code under [the repo for that
project](https://github.com/tom-f-oconnell/stimuli).

