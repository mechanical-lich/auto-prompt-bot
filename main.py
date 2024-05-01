import sys
import time
import schedule
from artist import Artist
import configparser



count = 1
artist = None

# Called by the scheduler - Triggers the artist to make art.
def job():
    global count
    global artist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("Attempt {count} at {timestamp}\n".format(count=count, timestamp=timestamp))

    artist.make_art(timestamp)
    count = count + 1

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    artist = Artist(
        sd_model_id = config["Models"]["sd_model_id"],
        llm_model_id = config["Models"]["llm_model_id"],
        llm_file_name = config["Models"]["llm_file_name"],
        num_inference_steps = int(config["Settings"]["num_inference_steps"]),
        seed_prompt = config["Settings"]["seed_prompt"],
        negative_prompt = config["Settings"]["negative_prompt"],
    )

    job() # Manually run the first job to get us started quick
    schedule.every().hour.at(":00").do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)