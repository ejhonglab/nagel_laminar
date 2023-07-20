#!/usr/bin/env python3
# TODO may want to  make a venv to install this py3 stuff, and link to that specific
# venv python in shebang line?

import atexit
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path as _Path
from tqdm import tqdm
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
# TODO delete
import matplotlib.patches as patches


def roi_yaml_path2roi_num(roi_yaml: Path) -> int:
    parts = roi_yaml.stem.split('_')
    roi_num_part = parts[1]
    assert roi_num_part.startswith('N')
    return int(roi_num_part[1:])


def make_path(roi: list):
    """
    Args:
        roi: should not already duplicate the end point
    """
    # TODO TODO TODO whatever solution i settle on, test independent of order. otherwise
    # maybe use convex hull stuff / shapely operations?

    # hourglass shape
    '''
    codes = [
        _Path.MOVETO,
        ] + (len(roi) - 1) * [_Path.LINETO] + [
        _Path.CLOSEPOLY
    ]
    return _Path(roi + [roi[0]], codes=codes)
    '''
    print(f'(inside make_path) {roi=}')

    # doesn't include one of the points
    #return _Path(roi, closed=True)

    # makes an hourglass shaped thing
    #return _Path(roi + [roi[0]], closed=True)
    return _Path(np.vstack([roi, roi[0]]), closed=True)


def put_text(image, text, position, font=cv2.FONT_HERSHEY_PLAIN, scale=1.0,
    color=(0, 0, 255), thickness=2, center_x=False):
    """
    Args:
        position: either (x, y) tuple, or y coordinate (if `center_x=True`)
    """
    if not center_x:
        xy = position
    else:
        # So we can center the text.
        (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
        # (0,0) is at the top left of the image.
        # This should be the bottom left of the text.
        #
        # It does actually seem it's (y,x), not (x,y).
        # TODO does that mean other code is wrong?
        xy = ((image.shape[1] - text_width) // 2, position)

    cv2.putText(image, str(text), xy, font, scale, color, thickness)


# (move this comment to bag2vid)
# TODO TODO is bag2vid shrinking width / height by 2 pixels each?  comparing size of
# camera and pngs to size of .avis seems so... is it already smaller in ROS delta vid?

# TODO move (part of?) this to multi_tracker repo (some is nagel_laminar specific...)

#@profile
def main():
    dupe_time_to_use = 'first'
    #dupe_time_to_use = 'last'

    # pixel distance a point can be outside of ROI bounds while still being
    # considered inside
    within_roi_tolerance_px = 5.0

    # TODO TODO TODO actually print fraction of points this applies too (maybe
    # previously filtering stuff outside ROI?)
    # TODO i didn't already have some other threshold for this, did i?
    #dist_between_points_thresh_px = 10

    make_plots = False

    # TODO revert to None
    check_roi = None
    #check_roi = 4

    debug = False

    if check_roi is not None:
        debug = True

    if debug:
        logging.warning('debug=True, so not saving annotated frames to .avi')

    # TODO two thresholds on dist from ROI:
    # 1) to fillna the points (drop the original ones altogether)
    # 2) to color stuff as being outside of ROI / show dist

    # Some bad Zihang data. LEDs may be fluctuating.
    #data_dir = Path('~/nagel_laminar/choice_20230705_164422').expanduser()
    # Good Betty data (with its own problems...).
    #data_dir = Path('~/nagel_laminar/20210325_123208').expanduser()
    data_dir = Path.cwd()

    # TODO accept argument for roi num and just load that one (for playing around with
    # data in a debugger)?

    use_filter_output = False
    if use_filter_output:
        # TODO TODO TODO try using measurment_x|y (should be unfiltered) instead of
        # position_x|y, on new data (re-run hdf5_to_csv.py w/ new -a flag)
        # TODO TODO maybe show both measurement and position?
        # TODO something in filename indicating measurement/position input?
        xy_cols = ['position_x', 'position_y']
    else:
        # NOTE: hdf5_to_csv.py need to be run with new -a/--all flag to also output
        # these columns
        xy_cols = ['measurement_x', 'measurement_y']

    tz = 'US/Pacific'

    avi_files = list(data_dir.glob('*.avi'))
    avi = None
    if len(avi_files) > 0:
        avi_files = [x for x in avi_files if 'tracking_check' not in x.name]
        assert len(avi_files) == 1
        avi = avi_files[0]

    # TODO also load .bag and check frames w/ no fly detected (need to save extra data
    # for this? put behind debug flag?), to see if threshold needs to be lowered?
    leave = False

    roi_num2roi = dict()
    roi_num2df = dict()
    roi_num2hull = dict()

    roi_yamls = data_dir.glob('roi_N*.yaml')
    for roi_yaml in roi_yamls:
        roi_data = yaml.safe_load(roi_yaml.read_text())['roi_points']

        roi_num = roi_yaml_path2roi_num(roi_yaml)

        if check_roi is not None and roi_num != check_roi:
            continue

        print(f'roi: {roi_data}')

        csvs = list(data_dir.glob(f'*_N{roi_num}_trackedobjects.csv'))
        assert len(csvs) == 1
        csv = csvs[0]

        df = pd.read_csv(csv)

        # TODO TODO TODO the zero here should be from the first tracking across all ROIs
        # (or else draw time in an ROI specific manner / only use time_epoch)
        df['time_s'] = df.time_epoch - df.time_epoch[0]
        #df['time'] = pd.to_datetime(df.time_epoch, unit='s')

        hull = cv2.convexHull(np.array(roi_data))
        roi_num2hull[roi_num] = hull

        roi = Polygon(hull.squeeze())

        # TODO vectorize this and/or the contains/distance calcs
        # TODO maybe make a Polygon then get Points from there?
        # 33% of time
        points = [Point(x) for x in tqdm(df[xy_cols].values, leave=leave,
            desc='constructing points', unit='point'
        )]

        # It seems that even with large tolerances, neither shapely nor matplotlib way
        # indicates many more points are within ROI (when drop_duplicates above is done
        # w/ keep='first')

        # NOTE: boundary points (e.g. roi corners themselves) are not considered as
        # being contained inside polygon. Would need to dilate polygon if i wanted to
        # allow some leeway here (or maybe a diff call handles boundary itself
        # differently, if that's all the behavior I'd want to change).
        # 40% of time
        # TODO TODO check against other (matplotlib?) method of checking whether a point
        # is inside a polygon: https://stackoverflow.com/questions/36399381
        # TODO TODO TODO TODO are x,y swapped between ROI and tracking (x,y) data???
        contained = [roi.contains(x) for x in tqdm(points, leave=leave,
            desc='checking points in ROI', unit='point'
        )]
        df['within_roi'] = contained
        # TODO delete (trying to match matplotlib results, which seem correct from
        # video)
        #df['within_roi3'] = [roi.exterior.contains(x) for x in points]

        # TODO TODO replace above calculation w/ this? this seems to be faster.
        # (if i could remove shapely entirely, it'd probably be faster)
        #arr = df[xy_cols].values
        ## radius kwarg available for tolerance on contains_points
        #contained2 = make_path(hull.squeeze()).contains_points(arr)
        #df['within_roi2'] = contained2

        # Could compute these just for stuff where roi.contains(x) is False
        # (and then check within a tolerance)
        roi_exterior = roi.exterior
        # 25% of time
        # TODO TODO sanity check these distance calculations? plot against video and
        # watch?
        df['distance_to_roi'] = [
            float('nan') if in_roi else roi_exterior.distance(x)
            for x, in_roi in tqdm(zip(points, contained), leave=leave,
                desc='calculating distances to ROI', unit='point'
        )]

        '''
        if roi_num == 4:
            # this ROI had the track diverge from ROI starting here
            # in keep='first' run on betty's 20210325_123208 data
            #t0 = 1616701023

            # this ROI had the track diverge from fly starting here
            # in keep='last' run on betty's 20210325_123208 data
            t0 = 1616701038
            # NOTE: in the same output, roi=1 has fly get lost (picks up a dark spot in
            # chamber that doesn't move) start at 1616701481-2

            pdf = df[df.time_epoch >= t0].copy()
            pdf.time_epoch = [str(x) for x in pdf.time_epoch]
            print(pdf.head(200).to_string())
            import ipdb; ipdb.set_trace()
        '''

        # TODO TODO TODO choice_20230705_164422/4 why does tracker lose fly at
        # t= ~1688601795.34? other track available (i.e. is it a data association error
        # recoverable without retracking)?
        if check_roi:
            import ipdb; ipdb.set_trace()

        # TODO TODO TODO option to not do this at all, to plot all ROIs below on the
        # movie (then maybe we could save data w/o retracking by just redoing the data
        # association step)
        # TODO TODO try dropping stuff outside of ROI *before* this (potentially needing
        # to then ffill for timestamps dropped completely), so that if it's not always
        # consistently first/last that is good, we get the best answer per timestamp
        #
        # NOTE: assuming first listed point is best track for that time point
        # (may not be true...)
        # Tried keep='last', to see if that reduces number out of ROI.
        # (it does, from ~80% out to 0% out. idk if some other part of tracking is less
        # good...)
        # TODO TODO draw both first/last on video and call out regions where they
        # differ, to compare
        df = df.drop_duplicates(subset='time_epoch', keep=dupe_time_to_use)

        # After drop_duplicates on time, no longer relevant.
        """
        # where are these coming from???
        #
        # (puts a NaN for first element)
        df['time_since_last_frame_s'] = df.time_s.diff()

        dt_zero = np.isclose(df.time_since_last_frame_s, 0)
        print(f'{dt_zero.sum() / len(df):.4f} fraction of rows with no time since prior'
            ' frame'
        )
        print('time delta (s) between rows in CSV (~frames):')
        print(df.time_since_last_frame_s.value_counts().to_string(header=False))
        """

        diff = np.diff(df[xy_cols], axis=0, prepend=float('nan'))
        assert len(diff) == len(df)
        assert diff.shape[1] == 2

        dists = np.sqrt((diff ** 2).sum(axis=1))
        assert len(dists) == len(diff)
        assert len(dists.shape) == 1

        # TODO TODO try only calculating (and including in histogram) distances where
        # neither point is outside of bounds (by more than some margin, at least)

        df['dist_between_points'] = dists

        # TODO convert to mm/s for these plots?

        zero_dists = np.isclose(df.dist_between_points, 0)
        print(f'{zero_dists.sum() / len(df):.4f} fraction of zero distances')

        #dist_between_points_thresh_px 

        log_x_scale = False
        # TODO return True
        log_counts = False

        if make_plots:
            dists_to_hist = df[~ zero_dists]

            fig, ax = plt.subplots()
            sns.histplot(dists_to_hist, x='dist_between_points', log_scale=log_x_scale,
                binwidth=0.25
            )
            if log_counts:
                ax.set_yscale('log')

            ax.set_title('distances between adjacent points (excluding 0)')

        # TODO this handling NaN in df.distance_to_roi correctly?
        not_within_roi = ~df.within_roi & (df.distance_to_roi > 0)
        print(f'{(~not_within_roi).sum() / len(df):.4f} fraction exactly within ROI')

        # (it seems to be the same)
        #not_within_roi2 = ~df.within_roi2 & (df.distance_to_roi > 0)
        #print(f'{(~not_within_roi2).sum() / len(df):.4f} fraction exactly within'
        #    ' ROI (2)'
        #)

        # TODO TODO some way to draw the specific lines corresonding to these distance
        # (on the cv2 plots below)?
        if make_plots:
            # TODO express as fraction of pixel dist from edge of ROI to center of ROI
            # (or to ROI long dimension or something?)
            roidistances_to_hist = df[not_within_roi]

            fig, ax = plt.subplots()
            sns.histplot(roidistances_to_hist, x='distance_to_roi',
                log_scale=log_x_scale
            )
            if log_counts:
                ax.set_yscale('log')

            ax.set_title('distances to ROI (excluding points inside ROI)')

        # TODO TODO calc size of ROI + frame rate, to get a sense of what reasonable
        # speeds are? frame rate just from time delta?

        # TODO TODO plot any of these things over time? (prob also need log scale)
        # TODO cumulative / sum?


        # TODO TODO plot walking speed over time (more to check flies rather than
        # tracker per se). was the long acclimation used in dhruv's paper ("~80-120
        # min") really necessary? what about doing stuff near subjective dusk "

        '''
        print('head:')
        print(df.head(1000).to_string())
        print('tail:')
        print(df.tail(1000).to_string())
        '''

        # NOTE: see alignment issues discussed below, in video drawing code
        assert df.time_epoch.is_monotonic_increasing
        # (zero-indexed)
        df['tracking_frame'] = df.groupby('time_epoch').ngroup()
        assert df.tracking_frame.is_monotonic_increasing

        # TODO TODO TODO how long are runs of missing (or crazy, way out of ROI) points?

        # TODO TODO plot them over time
        """
        # TODO TODO is it always only 2 rows w/ same timestamp? occurence of these
        # duplicates over time?
        with_dupes = df.groupby('time_epoch').filter(lambda x: len(x) > 1)
        dupe_counts = with_dupes.groupby('time_epoch').size().to_frame('count')
        if make_plots:
            fig, ax = plt.subplots()
            sns.histplot(dupe_counts, x='count', ax=ax)

        # TODO TODO color each separately in cv2 drawing below?
        '''
        for time_epoch, gdf in with_dupes.groupby('time_epoch'):
            print()
            print(f'{time_epoch=}')

            imin = gdf.index.min()
            imax = gdf.index.max()
            print('before:')
            df.loc[imin-2:imin]

            print(gdf)

            print('after:')
            df.loc[imax-1:imax+3]
            import ipdb; ipdb.set_trace()

        import ipdb; ipdb.set_trace()
        '''
        """

        if make_plots:
            plt.show()
            import ipdb; ipdb.set_trace()

        roi_num2roi[roi_num] = roi_data
        roi_num2df[roi_num] = df

        # TODO delete
        #print('SKIPPING OTHER ROIS!')
        #break
        #

    warned_tracking_time = False

    if avi is not None:
        cap = cv2.VideoCapture(str(avi))

        # TODO embed filtering parameters in name (e.g. 'keeplast')?
        filt_str = '' if use_filter_output else '_no-filter'
        output_name = f'tracking_check_keep{dupe_time_to_use}{filt_str}.avi'
        output_avi = data_dir / output_name

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not debug:
            # TODO overwrite flag?
            # TODO TODO TODO save in same directory as input
            # TODO uncomment
            #if output_avi.exists():
            #    raise IOError(
            #    f'output avi {output_avi} already exists! rename / delete'
            #)

            # Should basically be same as:
            # 1 / df.time_since_last_frame_s[df.time_since_last_frame_s > 0
            #    ].[mean|median]()
            fps = cap.get(cv2.CAP_PROP_FPS)

            format_code = 'XVID'
            fourcc = cv2.VideoWriter_fourcc(*format_code)
            out = cv2.VideoWriter(str(output_avi), fourcc, fps,
                (frame_width, frame_height)
            )
            atexit.register(out.release)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=n_frames, desc=f'writing {"/".join(output_avi.parts[-2:])}',
            unit='frame'
        )

        roi_num2tracking_frame_offset = dict()

        # TODO sequence of at least 4 good colors (+assert we have enough colors for
        # each?)
        # TODO or maybe just color seperately if there *are* other points, but just
        # draw the first one (keeping only first in each group, like betty did)
        # NOTE: cv2 uses bgr i think
        good = (0, 255, 0)
        # TODO use markers other than color of outer circle for these, so color sequence
        # can be used for diff good ones at less risk of confusion
        #large_diff = (255, 0, 0)
        # (orange)
        #out_of_bounds = (255, 125, 0)

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            curr_tracking_time = None

            for roi_num, roi in roi_num2roi.items():

                # TODO TODO TODO TODO why does track show up as not in ROI for
                # choice_20230705_164422/roi1 (starting ~1688600740.55)?
                # (doesn't seem to be unique to shapely calculation. matplotlib seems to
                # give the same result. (x,y) flipped somewhere?)
                #
                # TODO delete
                """
                if roi_num == 1:
                    df = roi_num2df[roi_num]
                    hull = roi_num2hull[roi_num]

                    odf = df[~df.within_roi]

                    fig2, ax2 = plt.subplots()
                    # TODO TODO TODO why does this show nothing?
                    #roi_path2 = make_path([list(x) for x in hull.squeeze()])
                    roi_path2 = make_path(hull.squeeze())
                    patch2 = patches.PathPatch(roi_path2, facecolor='red', lw=2)
                    ax2.add_patch(patch2)
                    ax2.set_xlim(0, frame_width)
                    ax2.set_ylim(0, frame_height)
                    ax2.scatter(*odf[xy_cols].iloc[0])

                    plt.show()
                    import ipdb; ipdb.set_trace()
                """
                #

                hull = roi_num2hull[roi_num]
                cv2.drawContours(frame, [hull], -1, (255, 0, 0))

                df = roi_num2df[roi_num]

                # assuming tracking and movie recording and at essentially the same
                # time, as seems to be the case. they do not start at the same time.
                if roi_num not in roi_num2tracking_frame_offset:
                    tracking_frame_offset = n_frames - df.time_epoch.nunique()
                    assert tracking_frame_offset > 0
                    roi_num2tracking_frame_offset[roi_num] = tracking_frame_offset
                else:
                    tracking_frame_offset = roi_num2tracking_frame_offset[roi_num]

                # TODO off-by-one?
                if i < tracking_frame_offset:
                    continue

                # TODO off-by-one?
                rows = df.loc[df.tracking_frame == (i - tracking_frame_offset)]

                #roi_curr_tracking_times = set(rows.time_s)
                roi_curr_tracking_times = set(rows.time_epoch)
                assert len(roi_curr_tracking_times) == 1
                roi_curr_tracking_time = roi_curr_tracking_times.pop()

                if curr_tracking_time is None:
                    curr_tracking_time = roi_curr_tracking_time
                else:
                    # TODO delete try/except
                    try:
                        assert roi_curr_tracking_time == curr_tracking_time
                    except:
                        tracking_time_diff = roi_curr_tracking_time - curr_tracking_time

                        # TODO TODO fix (in both branches below, ideally)!
                        if abs(tracking_time_diff) < 0.2:
                            if not warned_tracking_time:
                                logging.warning('tracking time differed by '
                                    f'{tracking_time_diff:.3f}s'
                                )
                                warned_tracking_time = True
                        else:
                            print(f'{roi_curr_tracking_time=}')
                            print(f'{curr_tracking_time=}')
                            print(f'{tracking_time_diff=}')
                            import ipdb; ipdb.set_trace()

                assert len(rows) == 1
                # Now that there is the drop_duplicates call above (on the time)
                # this should no longer happen.
                #if len(rows) > 1:
                #    print()
                #    print(f'multiple rows at frame={i}')
                #    print(rows)
                #    # TODO does this only get triggered later?
                #    # presumably it should get triggered at some point?
                #    import ipdb; ipdb.set_trace()

                for j, (_, row) in enumerate(rows.iterrows()):
                    x, y = row[xy_cols]
                    x = int(x)
                    y = int(y)

                    # TODO iterate over stuff sharing the same timestamp (diff colors?
                    # do they actually correspond across neighboring groups w/ same
                    # number of rows?)

                    cv2.circle(frame, (x, y), 8, good, 1)

                    # TODO TODO TODO fix! in choice_20230705_164422, roi 1, only a few
                    # seconds into video (and often throughout first few minutes)
                    # (also ROI 2-4 starting ~4:40), tracks are marked as out of ROI
                    # even though in many cases they are within ROI
                    if not row.within_roi:
                        put_text(frame, f'to ROI: {row.distance_to_roi:.1f}',
                            (x + 6, y - 10), scale=1.0, color=(0, 0, 255), thickness=2
                        )

                    if j > 0:
                        put_text(frame, j, (x + 6, y - 6), scale=1.0, color=(0, 0, 255),
                            thickness=2
                        )

            if curr_tracking_time is not None:
                time_y = int(frame.shape[0] * 0.1)
                put_text(frame, f'{curr_tracking_time:.2f}', time_y, scale=2.0,
                    color=(255, 255, 255), thickness=4, center_x=True
                )

            # TODO flag to still enable this (/ switch from writing)?
            if debug:
                cv2.imshow('frame', frame)
            else:
                out.write(frame)
                pbar.update()

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break

            i += 1

        if not debug:
            out.release()


if __name__ == '__main__':
    main()

