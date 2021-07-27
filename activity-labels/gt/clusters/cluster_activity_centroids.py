"""
DESCRIPTION
-----------
The following script creates unsupervised clusters of activity
centroids.

USAGE
-----
python cluster_activity_centroids.py <activity csv> <img>

EXAMPLE
-------
```bash
# Linux
python cluster_activity_centroids.py /home/vj/Dropbox/typing-notyping/C1L1P-E/20170302/gTruth-tynty_30fps.csv /home/vj/Dropbox/typing-notyping/C1L1P-E/20170302/G-C1L1P-Mar02-E-Irma_q2_04-08_30fps.png 5
```
"""
import os
import pdb
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# User defined libraries
import pytkit as pk


def _arguments():
    """Parses input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            DESCRIPTION
            -----------
            The following script creates unsupervised clusters of activity
            centroids.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        "activity_csv",
        type=str,
        help=("Ground truth CSV file having activities")
    )
    args_inst.add_argument(
        "img",
        type=str,
        help=("An image of the session.")
    )

    args_inst.add_argument(
        "nc",
        type=int,
        help=("Number of clusters")
    )
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'activity_csv': args.activity_csv,
                 'img': args.img,
                 'nc': args.nc
                 
                 }

    # Return arguments as dictionary
    # Hello world how are you doing
    return args_dict



def get_centroids(df):
    """ Returns a dataframew with one centroid for one second
    of activity.

    Parameters
    ----------
    df : DataFrame
        A data frame having activity instances
    
    Returns
    -------
    DataFrame
        A data frame having centroids with one row corresponding to
        one second of activity.
    """
    
    # wc = centroid coordinate along width, hc = centroid coordinate
    # along height
    centroids = []
    columns = ["wc", "hc"]

    # Loop through each instance
    for i, row in df.iterrows():

        wc = int(row['w0'] + (row['w']/2))
        hc = int(row['h0'] + (row['h']/2))

        centroids += [[wc, hc]]

    return pd.DataFrame(centroids, columns=columns)
    
    
    

    
def main():
    argd = _arguments()
    if pk.check_file(argd['activity_csv']):
        df = pd.read_csv(argd['activity_csv'])

    # Calculating centroids dataframe for each activity
    centroid_df = get_centroids(df)

    # Fit K-Means clusters, due to 5 people
    K = argd['nc']
    kmeans = KMeans(n_clusters=K).fit(centroid_df)
    centroids = kmeans.cluster_centers_

    # Show plot
    im = plt.imread(argd['img'])
    implot = plt.imshow(im)
    plt.scatter(
        centroid_df['wc'],
        centroid_df['hc'],
        c= kmeans.labels_.astype(float),
        s=100,
        alpha=0.75
    )
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)

    # Title
    plt.title(f"Kmeans clustering with K = {K}", fontsize="20")

    # Playing with axis
    plt.axis('off')

    # Save the image
    out_file_name = f"{os.path.splitext(argd['img'])[0]}_cluster_K{K}.png"
    plt.savefig(out_file_name, bbox_inches='tight')

# Execution starts here
if __name__ == "__main__":
    main()
