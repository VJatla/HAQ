import argparse
from aqua.act_maps_and_perf import ActMapsAndPerf
            
def _arguments():
    """ Parses input arguments """
    
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("""
        Evaluates performance and viusalize it via activity maps.
        The input directory should have the following files,
            1. gt-ty-30fps.xlsx       : Excel file having typing instances from ground truth.
            2. alg-ty-30fps.xlsx      : Excel file having typing instances from algorithm.
            3. properties_session.csv : Session properties
            4. groups_jun28_2021.csv  : Groups database from AOLME website
                                        
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("idir", type=str,help="Input directory having necessary files")
    args_inst.add_argument("odir", type=str,help="Output directory")
    args_inst.add_argument("vdb", type=str,help="Video database")
    args_inst.add_argument("act", type=str,help="Activity name")
    args = args_inst.parse_args()

    args_dict = {
        'idir': args.idir,
        'odir': args.odir,
        'vdb' : args.vdb,
        'act' : args.act
    }
    return args_dict


# Execution starts from here
if __name__ == "__main__":
    
    # Initializing Activityperf with input and output directories
    args = _arguments()
    ty_perf = ActMapsAndPerf(**args)

    # Visualize activity maps
    # ty_perf.visualize_maps()


    # Write the map produced by confusion matrix
    ty_perf.write_maps(args['odir'])

    # Prints confusion matrix per person to a csv file
    # output_pth = f"{args['odir']}/cf_mat_per_person.csv"
    # ty_perf.save_cf_mat_per_person(output_pth)
    
