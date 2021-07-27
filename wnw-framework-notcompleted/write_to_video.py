"""
<Script description goes here>
"""
import argparse
import pandas as pd
import cv2
import aqua

def write_to_video(vpath, df):
    """
    Writes a video with writing/nowriting labeled. The video
    has name <video name>_w_using_alg.avi

    Todo
    ----
    - Written in a hurry clean it up.
    """
    vid  = aqua.video_tools.Vid(vpath)
    
    # Fonts
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (175,375)
    fontScale              = 1
    red_fontColor              = (0,0,255)
    blue_fontColor              = (255,0,0)
    green_fontColor              = (0,255,0)
    lineType               = 2

    vname = vid.props['name']
    vloc = vid.props['dir_loc']
    out_vid_pth = f"{vloc}/{vname}_w_using_alg.avi"

    out = cv2.VideoWriter(out_vid_pth,cv2.VideoWriter_fourcc('M','J','P','G'),
                          30,
                          (vid.props['width'],vid.props['height']))
    # loop over writing data frame
    for i, row in df.iterrows():
        for poc in range(int(row['f0'].item()), int(row['f0'].item() + row['f'].item()), 10):
            frm = vid.get_frame(poc)
            print(poc)

            if row['writing'] == -1:
                frm = cv2.putText(frm,'Hands not detected', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            red_fontColor,
                            lineType)

            elif row['writing'] == 0:
                frm = cv2.putText(frm,'no writing',
                                  bottomLeftCornerOfText, 
                                  font, 
                                  fontScale,
                                  blue_fontColor,
                                  lineType)
                frm = cv2.rectangle(frm,
                                    (int(row['w0']), int(row['h0'])),
                                    (int(row['w0'] + row['w']), int(row['h0'] + row['h'])),
                                    blue_fontColor,
                                    3
                                    )
            elif row['writing'] == 1:
                frm = cv2.putText(frm,'writing', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            green_fontColor,
                            lineType)
                frm = cv2.rectangle(frm,
                                    (int(row['w0']),int(row['h0'])),
                                    (int(row['w0'] + row['w']),int(row['h0'] + row['h'])),
                                    green_fontColor,
                                    3
                )
            else:
                raise Exception(f"{row['writing']} label not supported")
            out.write(frm)

    out.release()
    
def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Script description goes here
        """))

    # Adding arguments
    args_inst.add_argument("vpth", type=str, help=("<pos arg help>"))
    args_inst.add_argument("wcsv", type=str, help=("<pos arg help>"))
    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'vpth': args.vpth,
                 'wcsv': args.wcsv}

    # Return arguments as dictionary
    return args_dict


def main():
    """ Main function """
    argd = _arguments()
    df = pd.read_csv(argd['wcsv'])
    write_to_video(argd['vpth'],df)

# Execution starts here
if __name__ == "__main__":
    main()
