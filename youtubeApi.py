
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
import re
import os



class YoutubeApi:
    def __init__(self,API_KEY,SECRET_KEY) -> None:
        
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.youtube = None
        self.channel_id = None
        self.transcript = None
        self.user_id = None
        self.cleaned_text = None
        self.create()

    def create(self):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

        flow = InstalledAppFlow.from_client_secrets_file(
            self.SECRET_KEY, SCOPES)
        credentials = flow.run_local_server()
        self.youtube =  build(
            "youtube", 'v3', credentials=credentials)
        
    def get_channel_id(self,username):

        #youtube = build('youtube','v3',developerKey=API_KEY)
        request = self.youtube.channels().list(
            part='id',
            forUsername=username
        ).execute()

        if request['pageInfo']['totalResults'] == 0:
            return False

        self.channel_id = request['items'][0]['id']


    def get_channel_videos(self):
        video_resp = self.youtube.search().list(
            part='id',
            channelId=self.channel_id,
            type='video'
        ).execute()

        video_id = [item['id']['videoId'] for item in video_resp['items']]
        return video_id
    


    def save_document(self):
        self.extract_text_from_transcript()
        with open("saves.txt",'w') as doc:
            doc.write(self.cleaned_text)
            doc.close()

    def get_video_transcript(self,video_id):

        captions = self.youtube.captions().list(
            part='snippet',
            videoId=video_id
        ).execute()

        if 'items' in captions:
            # Videonun bir veya daha fazla altyazısı varsa
            caption_id = captions['items'][0]['id']

            # Altyazı metnini indir
            caption_text = self.youtube.captions().download(
                id=caption_id
            ).execute().decode('utf-8')

            self.transcript = caption_text

            return caption_text
        
        else:

            return None
        
    def get_video_title(self,video_id):

        # Videos.list API'sini kullanarak video bilgilerini al
        request = self.youtube.videos().list(
            part='snippet',
            id=video_id
        )
        
        response = request.execute()
        
        # Response'dan video başlığını çıkar
        if 'items' in response:
            title = response['items'][0]['snippet']['title']
            return title
        else:
            return None

    def extract_text_from_transcript(self):
        # Zaman damgalarını içermeyen metni bulmak için regular expression kullanma
        pattern = re.compile(r'\d+:\d+:\d+\.\d+,\d+:\d+:\d+\.\d+\n(.*?)\n', re.DOTALL)
        
        # Regular expression ile eşleşen tüm metinleri bul
        matches = pattern.findall(self.transcript)
        
        # Eşleşen metinleri birleştir ve temizle
        cleaned_text = ' '.join(matches)
        cleaned_text = cleaned_text.replace('\n', ' ').strip()

        self.cleaned_text = cleaned_text



        

if __name__ == '__main__':

    """
    for i in range(len(video_ides)):
        try:
            
            
        except Exception as e:
            print("sagibinin videolar için indirmeye izin vermesi gerekir hata :",e)
            pass

    print(Api.transcript)
   
    """