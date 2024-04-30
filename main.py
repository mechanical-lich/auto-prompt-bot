from diffusers import AutoPipelineForText2Image
from PIL import Image
import sys
import time
import torch
import schedule
from llama_cpp import Llama
from artist import Artist

count = 1
artist = None

def job():
    global count
    global artist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("Attempt {count} at {timestamp}\n".format(count=count, timestamp=timestamp))

    artist.make_art(timestamp)
    count = count + 1

if __name__=="__main__":
    artist = Artist()
    job()
    schedule.every().hour.at(":08").do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)