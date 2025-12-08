The code entry for the single graphics card of consisid is teacache_inference_consisid.py. It directly uses the consisid model integrated into the diff library. 
If modifications are needed, they should be made directly in the library.

Before using the code, you need to download the ckpts of consisid and all its running environments 
(see requirements.txt and ckpts/README.md and README_ConsisID.md).Before using the code, you need to set up a Baidu large model account and its API key, 
and fill the key into line 11 of qianfan.py.Examples of code generation can be found in finalresult.

The code runs through the terminal. Example of running: 
python auto_run.py --image "ConsisID-data\eval\face_images\1_stars_woman_Taylor_Swift_1.png" --prompt "A woman with a genuine smile tilts her head slightly toward the camera, her eyes reflecting the soft glow of the golden hour as the urban skyline forms a majestic backdrop; in a moment of spontaneity, a gentle breeze tousles her hair, adding a sense of vibrant energy to the serene atmosphere." --save_path "result4" --loop_num 3

Execute python auto_run.py to see the help.