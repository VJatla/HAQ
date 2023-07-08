"""
Description
-----------
    Extract frames from writing and no-writing samples to hands-pen and hands-nopen
    directories.

Output
------
    A directory that has frames extracted to output directory. They are placed in `hands_with_pen`
    and `hands_without_pen` directories.

Example
-------
python extract_frames.py \
    /home/vj/twotb/aolme_datasets/wnw_table_roi/hands_with_withot_pen_images/writing_30fps \
    /home/vj/twotb/aolme_datasets/wnw_table_roi/hands_with_withot_pen_images/nowriting_30fps \
    /home/vj/twotb/aolme_datasets/wnw_table_roi/hands_with_withot_pen_images

"""


import argparse
import os
import pytkit as pk
import cv2


def _arguments():

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """Extract frames from writing and no-writing samples to hands-pen and hands-nopen
            directories.
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("writing_dir", type=str, help="Directory containing writing samples.")
    args_inst.add_argument("nowriting_dir", type=str, help="Directory containing nowriting samples.")
    args_inst.add_argument("out_dir", type=str, help="Output directory")
    args = args_inst.parse_args()

    args_dict = {
        "writing_dir": args.writing_dir,
        "nowriting_dir": args.nowriting_dir,
        "out_dir": args.out_dir
    }
    return args_dict


def extract_frames(idir, odir, frames_to_extract = 1):
    """Extracts `frames_per_sec` frames from video every second

    Parameters
    ----------
    idir : Str
        Input directory having videos
    odir : Str
        Output directory to place the extracted frames
    frames_per_sec : int, optional
        Number of frames to extract. Defaults to 1.
    """
    vpaths = pk.get_file_paths_with_kws(idir, ['.mp4'])
    vnames = [
        os.path.splitext(os.path.basename(x))[0] for x in vpaths
    ]
    for i in range(0, len(vpaths)):
        
        vpath = vpaths[i]
        vname = vnames[i]
        vid = pk.Vid(vpath, 'read')

        # Calculating equally spaced intervals between start and end of frames
        if frames_to_extract == 1:
            frame_idxs = [vid.props['num_frames']/2 - 1] 
        else:
            import pdb; pdb.set_trace()
            
        for frame_num in frame_idxs:
            oimg_path = f"{odir}/{vname}_{frame_num}.png"
            img = vid.get_frame(frame_num)
            cv2.imwrite(oimg_path, img)     

# Execution starts from here???
if __name__ == "__main__":

    # Parsing input arguments to variables
    args = _arguments()
    wrdir = args['writing_dir']
    nwrdir = args['nowriting_dir']
    odir = args['out_dir']

    # Extracting hands
    odir_hands_with_pen = f"{odir}/hands_with_pen"
    print(f"Extracting hands with pen to {odir_hands_with_pen}")
    if not os.path.isdir(odir_hands_with_pen):
        os.mkdir(odir_hands_with_pen)
    extract_frames(wrdir, odir_hands_with_pen, frames_to_extract = 1)

    # extracting hands without pen frames
    odir_hands_without_pen = f"{odir}/hands_without_pen"
    print(f"Extracting hands without pen to {odir_hands_without_pen}")
    if not os.path.isdir(odir_hands_without_pen):
        os.mkdir(odir_hands_without_pen)
    extract_frames(nwrdir, odir_hands_without_pen, frames_to_extract = 1)
    print(args)
