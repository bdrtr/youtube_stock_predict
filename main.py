from youtubeApi import YoutubeApi
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from ddi_model import vectorizer,CleanModel


if __name__ == '__main__':

    API_KEY = "AIzaSyDY6-qxG0j1A4meNJyyiMVDXyvmqS8MtZU"
    path_of_serv = '/home/bedir/Documents/vsCode3/python/python3/youutbeapi/client_secret_254761862433-6k3itj0ss4tno9ogmcincs0up58nj2d5.apps.googleusercontent.com.json'

    ben = 'UC5qrUS7Xvi--AcYUwOEI8YQ'
    Api = YoutubeApi(API_KEY=API_KEY,SECRET_KEY=path_of_serv)
    #Api.get_channel_id('@ParaBilgi')
    #Api.channel_id = 'UC5qrUS7Xvi--AcYUwOEI8YQ'
    #print(Api.channel_id == 'UCw1RjJTOmb5fjj7NuKT12Qw')
    #Api.channel_id = 'UCw1RjJTOmb5fjj7NuKT12Qw'
    Api.channel_id = ben
    video_ides = Api.get_channel_videos()
    print(video_ides)
    print(Api.get_video_title(video_ides[0]))
    print(Api.get_video_title(video_ides[1]))

    print(Api.get_video_transcript(video_ides[0]))
    Api.save_document()

    text = CleanModel.clean_text(Api.cleaned_text)
    with open('/home/bedir/Documents/vsCode3/python/python3/youutbeapi/clf_SVM2.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    res = loaded_model.predict(vectorizer.transform([text]))
    print(res)