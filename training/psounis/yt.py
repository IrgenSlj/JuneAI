from random import randrange

class Playlist:
    def __init__(self, title="", description="", time=0, videos: list = []):
        self.title = title
        self.description = description
        self.time = time
        self.videos = videos

    def add_video(self, video):
        self.videos.append(video)
        print("Video added")

    def __str__(self):
        st = f"\nPlaylist name: {self.title}"
        st += f"\nDescription: {self.description}"
        st += f"\nDuration: {self.time}"
        st += "\nVideos:"
        for video in self.videos:
            st += f"\nvideo: {video.artist_name}, {video.track_name}, {video.time}"
        return st

    def recommendation(self):
        rand = randrange(len(self.videos))
        return self.videos[rand].__str__() if rand >=0 else None

class Video:
    def __init__(self, artist_name, track_name, time):
        self.artist_name = artist_name
        self.track_name = track_name
        self.time = time

    def __str__(self):
        st = f"Video: {self.artist_name}, {self.track_name}"
        return st 
    
class Classic_pl(Playlist):
    def __init__(self, title, description, time, videos, period):
        super().__init__(title, description, time, videos)
        self.period = period

    def recommendation(self):
        return self.videos[2].__str__()

video1 = Video("MJ", "Dangerous", 240)
video2 = Video("Sheeran", "Love", 350)
video3 = Video("Extra", "ex5tra", 456)

pl = Playlist("Hits", "Very good", 3000, [video1, video2])

pl.add_video(video3)
print(pl.__str__())
print()
print(pl.recommendation())
print()

pl2 = Classic_pl("Barok", "Complex", 4500, [video1, video2],"Barok")
pl2.add_video(video3)

print(pl2.recommendation())