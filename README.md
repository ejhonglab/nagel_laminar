
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

