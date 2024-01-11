# Hisse senetlerinin ileri dönem hareketlerinin tahmini

Bu program içeriğindeki yardımcı kütüphaneler ve apiler sayesinde youtube'dan analiz videoları izleyerek yapılan yorumları değerlendirir ve aranan hisse hakkında yorumların ne olduğunu açıklar

## Kullanılan Kütüphaneler

- Pandas
- NumPy
- google
- googleapiclient
- Scikit-learn
- Nltk
- Pyplot



# VERİ SETİ'NİN OLUŞTURULMASI

Dataseti Kaggle ve elimle oluşturduğum verilerden oluşcaktır.Çıkarabilcek sonuçlar arasında hissenin gelecek dönemde olumlu, olumusuz veya belirsiz bir yol izleyeceği seçilcektir burda anahtar kelimeler ve verilerin temizliği uygunluğu model'in doğru çalışması için gereklidir.Bunun için ise aranan nitelikler şöyledir:

- Yorumlar argo içermemeli
- Kelimelerin doğru yazılması
- stop word'lerin olabildiğince az kullanılmış olması

Bu kriterleri sağlayan tüm veri setleri kaynak olarak kullanılabilir. 

Pythonda youtube ile istenilen verileri çekebilmek için farklı yetenekleri olan pyton kütüphanesi google ve googleapiclient kullanılıyor.
[apiler](https://github.com/googleapis/google-api-python-client)



Youtube apisinin bir videoyu çekebilmesi için transcipt'in oluşturulması (bu genelde her video için youtube tarafıdan otomatik olarak sağlanır) ikinci olarak ise videonun 3.taraf yazılımlar tarafından indirme izininin olması gerekir aksi halde video'ya indirmek isteği bir [permission denied](https://docs.github.com/en/authentication/troubleshooting-ssh/error-permission-denied-publickey)
istisnasına yol açar

### Örnek bir sorgu:
```
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
```

Bu sorgu transcipt için istekte bulunur

Youtube api kuralları gereği her bir yeni istek bir doğrulama ekranına yönlendirilir bu nedenle program kapatılıp açılması bir son kullanıcıya ihtiyaç duyar

![doğrulama ekranı](image/assets/dogrulama_ekran.png)

### Verilerin Çekilme Süreleri:

12 dk'lık bir videonun transcipt'inin inidirilme süresi 12 saniyedir.


### Oluşturulmuş Olan Veri seti'nin Görseli:
![data_set](bdrtr/youtube_stock_predict/issues/17)


# Yorumların Temizlenmesi ve Lemmatization İşlemi
Veriseti oluşturulduktan sonra modelin daha iyi çalışması ve başarı oranının daha yüksek olması için yorumların temizlenmesi gerekmektedir. Yorumların içerisinde emojiler, noktalama işaretleri, stopwordsler, linkler gibi istenmeyen ve modelin başarısını düşürecek veriler yorumlar içerisinden temizleniyor. Daha sonra lemmatization (kelimelerin köklerinin alınması) işlemi yapılarak temiz ve kelimelerin köklerinden oluşan yorumlar elde ediliyor.

### Oluşturulan Temiz Yorum Görseli
![clean](https://private-user-images.githubusercontent.com/69633060/293515242-34eae497-0a1e-434c-9291-fe4a157692b9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3Mzg2NjMsIm5iZiI6MTcwNDczODM2MywicGF0aCI6Ii82OTYzMzA2MC8yOTM1MTUyNDItMzRlYWU0OTctMGExZS00MzRjLTkyOTEtZmU0YTE1NzY5MmI5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MjYwM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ1NzM4NTM1YTlmOWE0NmNmMjM5OTAwZDEzOGJiMGY2YTdhMWU5NjdiNDQ3M2U0MDgxYzNjMTliNWZhZWM2OTImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.CA2MPp6EFYhC43refJZih5EdQsvDDfNojPE1Pjra9EE)


# Modelin Oluşturulması ve Yorumların Kategorilendirilmesi

## Model Seçimi
Yapılacak kategorilendirme işleminin hangi modelde daha yüksek başarı oranı vereceğini tespit etmek amacıyla araştırma yapılıp aynı zamanda bazı modeller üzerinde de test edilmiştir. Başlangıç olarak 3 popüler model üzerinde denemeler yapılmıştır. Bu modeller Naive Bayes, DecitionTree ve SVM modelidir. Veriseti üzerinde bu modellerin accuracy ve f1 score ları test edilmiştir. Projedeki test veriseti sonuçlarına bakıldığında:

- Naive Bayes Modeli için  f1 score: 0.710 
- DecitionTree Modeli için f1 score: 0.818
- SVM için (linear) f1 score: 0.765
- SVM (rbf) için f1 score: 0.807

![](https://private-user-images.githubusercontent.com/69633060/293515525-ace13de2-1094-4020-9a32-8b697e6db440.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkxNzEsIm5iZiI6MTcwNDczODg3MSwicGF0aCI6Ii82OTYzMzA2MC8yOTM1MTU1MjUtYWNlMTNkZTItMTA5NC00MDIwLTlhMzItOGI2OTdlNmRiNDQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzQzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWUyZmZmMzEzMjY2NWE1NWFiZmYxZjViZmYyNjQxZDI0MWQ4OWRkNzQyZmJjNzM1Zjk3ZDU4OGVjZDIyYTE3Y2YmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.kUxGyZCDKKaeEkfWgVrROr5dhaW6Sl2LcVGsSlsiJdw)

kullanılan dilin farklılığı ve teknik analiz kavramları doğruluk oranlarını oldukça düşürmektedir.


Sonuçlar incelendiğinde DecitionTree ve SVM (linear c=10) modelinin projede kullanılan verisetine göre vericeği sonuçların başarısı yeterlidir. Bu iki modelin kullanımına karar verilmiştir.

Support Vector Machine modeli kullanılmıştır. 

## Modelin Oluşturulmaya Başlanması

### Etiketleme
Yapılan yorumların bulunduğu "yorumlar.csv" dosyası dataframeye aktarılmıştır. Yorumların ifade ettiği değerlendirmeler -1 0 1 gibi bilgisayarın anlayabileceği bir formata dönüştürülmelidir. 'etiketli' adında yeni bir kolon açılarak olumsuz yorumlar için -1 sayısı, olumlu yorumlar için  1 sayısı, tarafsız yorumlar için 0 sayısı eklenmiştir.
 
![](https://private-user-images.githubusercontent.com/69633060/293516829-989cbcaa-00e0-4594-80f2-6205956a11ac.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkyMDksIm5iZiI6MTcwNDczODkwOSwicGF0aCI6Ii82OTYzMzA2MC8yOTM1MTY4MjktOTg5Y2JjYWEtMDBlMC00NTk0LTgwZjItNjIwNTk1NmExMWFjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzUwOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcyZjgzZTRhZTY3NzFlYmJlZDFlYWNhYzhjZjI2YmJiYTljNDNiY2Q2NDhiYWIyYzZjOGYzNDA0OGU4NDYxZDMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.L9Y9t6t0PfFpeEkqYW7JjPObaLXY3S_4ujuim98ggUU)

### Verisetinin Parçalanması
Modelin başarısını doğru şekilde ölçebilmek için modeli eğittiğimiz veriler ile test ettiğimiz veriler farklı olmalıdır. Modelin eğitilmiş olduğu verileri tekrar modele gönderirsek model bu veriler ile eğitildiği için başarısı yüksek ve yanıltıcı olacaktır. Bu yüzden verisetini parçalamamız gerekir. Bu projede verisetinin %80 'i modeli eğitmek için, %20 'si de modeli test etmek için kullanılacaktır.

Verisetini parçalamak için train_test_split() fonksiyonu kullanılmıştır.

### Tweetlerin Vektörel Matrisinin Çıkarılması
Yorumlar metinden oluştuğu için bunun bilgisayar ortamında işlenmesi mümkün değildir bu yüzden veriler sayısal değerlere dönüştürülmelidir. Bir sözlük oluşturularak dökümandaki her kelime için bir indexleme yapılır. Daha sonra hangi index numarasına sahip kelimenin hangi yorumda kaç kere geçtiği hesaplanarak sayma matrisi oluşturulur. Bu işlemi yaparken tf-idf vectorizer kullanılarak bir kelimenin döküman içindeki önemi istatistiksel olarak hesaplanmıştır. Bu sayede her tweette geçen model için anlamsız kelimelerin önemi düşürülmüştür yani stopwordsler tekrardan ayıklanmıştır.

### Modellerin Eğitilmesi
Daha önceden parçalanmış olan X_train ve y_train verileri Naive Bayes ve Support Vector Machine modeline gönderilerek modeller eğitilmiştir. Eğitim sonucunda modellerin accuracy ve f1 score değerleri hesaplanmıştır. Modelleri eğitmek için sklearn kütüphanesi kullanılmıştır.

### Modellerin Başarısının Hesaplanması
Modelin başarısı hem train hem test verileri üzerinden Accuracy ve F1 score ile ölçülmüştür. Alınan sonuçlar aşağıda bulunmaktadır.

Naive Bayes                |  Support Vector Machine  
:-------------------------:|:-------------------------: 
|![](https://private-user-images.githubusercontent.com/69633060/293516972-6d0e24f0-50ae-482f-91a0-86a897c64631.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkyNDMsIm5iZiI6MTcwNDczODk0MywicGF0aCI6Ii82OTYzMzA2MC8yOTM1MTY5NzItNmQwZTI0ZjAtNTBhZS00ODJmLTkxYTAtODZhODk3YzY0NjMxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzU0M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTRjODE1NjlhZWEwYjA0NWFhNzBjZTU4ZDczYjVjMTFmYzRkMjI3ZTcyY2VhYjQzZDBmMGE1ZmE3ZmZkNDAyODcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.tM0FhfKTkoiM8hdOO4Jh3WmWtP9DG4hzptNkNf68XD4)| ![](https://private-user-images.githubusercontent.com/69633060/293517014-f5b0258f-6712-45c6-996e-84518fb181b1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3Mzg1OTgsIm5iZiI6MTcwNDczODI5OCwicGF0aCI6Ii82OTYzMzA2MC8yOTM1MTcwMTQtZjViMDI1OGYtNjcxMi00NWM2LTk5NmUtODQ1MThmYjE4MWIxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MjQ1OFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThmYjVmYjE2ZmJlNjdmNDYzYjdhMDA4NDlhYWE2YTE0YWIxOTg3ZjlkNzViNWJmM2NiYWZkZjExMTllN2ViZTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.6eWlaIRVwcvFWV5DJOvznxBKde8RXz6MtEBr2P2i-nQ)  |

### Modellerin Hata Oranları
- Modelin inceliklerini anlamaya çalıştığımızda yorumlar keskinleşince bayes yanlış sonuç veriyor.

![inceleme](https://private-user-images.githubusercontent.com/69633060/293517244-673edce4-6458-439b-b504-9d70659a2306.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3Mzg1NjIsIm5iZiI6MTcwNDczODI2MiwicGF0aCI6Ii82OTYzMzA2MC8yOTM1MTcyNDQtNjczZWRjZTQtNjQ1OC00MzliLWI1MDQtOWQ3MDY1OWEyMzA2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MjQyMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM3NmMzMTE0ZmY5NzgwODEwYWFlOTA0MzViMjRjOThlMzcyZGE4NjkwYmIzZGE2MjI2YjkyYzdiN2JiZDc4MTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.-ghnoyR4eKpw2C8wG9ecgfHUvlI3_D-7qUx6-CvI0ow)

Modellerin hata oranlarını tespit etmek için 2 farklı metrik kullanılmıştır. Bunlar Ortalama Kare Hatası(MSE) ve Ortalama Mutlak Hata(MAE) dir.

##### Ortalama Kare Hatası(MSE)
Ortalama Kare Hatası tahmin edilen sonuçlarınızın gerçek sayıdan ne kadar farklı olduğuna dair size mutlak bir sayı verir.

##### Ortalama Mutlak Hata(MAE)
Ortalama mutlak hata, mutlak hata değerinin toplamını alır, hata terimlerinin toplamının daha doğrudan bir temsilidir.

![image](https://private-user-images.githubusercontent.com/69633060/294951792-53d04af1-9a31-4e50-b580-23f67ba0f994.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkyODgsIm5iZiI6MTcwNDczODk4OCwicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTE3OTItNTNkMDRhZjEtOWEzMS00ZTUwLWI1ODAtMjNmNjdiYTBmOTk0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzYyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM3ZGVlN2Y2ZDdiYzI3OTdjMTAzOWRiOGE2M2E3MzMwNTg2ZTNjNWY1ZDViY2Y2YTUwMzJmOTU4ZDVkYmM4NWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.gjzCTHtpSjHoVa5m5Vi_AXX8Bz7pyvXifrsOqCB0i4Q)

Sonuçlar incelendiğinde Support Vector Machine modelinin Naive Bayes modeline göre biraz daha az hata yaptığını görüyoruz.


### Manuel kontrol
Modeli manuel olarak test etmek için elle bazı yorumlar girilecek ve modelin bu yorumların hangi durumlara ait olduğunu tahmin etmesi istenecektir. Test sonuçları aşağıda gösterilmiştir.


##### Test1:
![test1](https://private-user-images.githubusercontent.com/69633060/294952974-ca985397-2d62-478b-9bd8-3fe72db8cc8b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkzMTQsIm5iZiI6MTcwNDczOTAxNCwicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTI5NzQtY2E5ODUzOTctMmQ2Mi00NzhiLTliZDgtM2ZlNzJkYjhjYzhiLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzY1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTU0ZjVkNTkyNjc1ZmE4ZWQzOTg0NWQ5OTRmOTc4OTM2YWM3ZGNhMTkyNzJkYTU5NTRiZTc1YmE0YzE1ZjAwYmMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.KVcHAs1yFka1sa4K_j30zjmefB_MIMfwwj_g1BGY6PU)

##### Test2:
![test2](https://private-user-images.githubusercontent.com/69633060/294953324-fe70999c-49fb-42d0-83d5-ded8c25f6d23.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkzMzIsIm5iZiI6MTcwNDczOTAzMiwicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTMzMjQtZmU3MDk5OWMtNDlmYi00MmQwLTgzZDUtZGVkOGMyNWY2ZDIzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzcxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcyMDgzZDdmYmZiYmQzMjIxZDY0NTFkY2VmY2ZlOGRmM2M2ZGZlN2U2N2RmNTQ0MjNiNzcxN2NiMzEwNjk5MDAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.RvkE8fEmFm2YbyW9szLTain7AuRajbG4pbayKFdc4b4)

##### Test3
![test3](https://private-user-images.githubusercontent.com/69633060/294953569-f08ce87c-54b7-4f44-8d3d-282be840b7bc.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkzNTMsIm5iZiI6MTcwNDczOTA1MywicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTM1NjktZjA4Y2U4N2MtNTRiNy00ZjQ0LThkM2QtMjgyYmU4NDBiN2JjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzczM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJlY2IxMmExODdhZTRjMjEzNDE2ODM2NjZlY2M5ZjA2MmU2OWM2NGNkMGMzY2NhNzdjNjc5ZGVhMjg5NDUwYjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.8EKWZ8Y2mo9r9Mfc9AmDYGuXjVSazi1zR420GI7jMQw)

##### Test4
![test4](https://private-user-images.githubusercontent.com/69633060/294953835-858169b5-ef1d-40c5-be5e-d9bd18c5ee96.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkzODEsIm5iZiI6MTcwNDczOTA4MSwicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTM4MzUtODU4MTY5YjUtZWYxZC00MGM1LWJlNWUtZDliZDE4YzVlZTk2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzgwMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJhNGJjZmI3NWFmMWI0ZDkxYjk0ZDIyYmYxNmZhZWUwMzQ5MDliZWMxNWI4OWZjZDE1YWNkYmQyOGRmZTUwNGMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.5HUAM9SCpuSBsYbr8j0dsl8S4wD_XS69mptKeVX0648)

##### Test5
![test5](https://private-user-images.githubusercontent.com/69633060/294954230-74c0b249-7752-4067-98e1-27ef89cf026b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzkzOTcsIm5iZiI6MTcwNDczOTA5NywicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTQyMzAtNzRjMGIyNDktNzc1Mi00MDY3LTk4ZTEtMjdlZjg5Y2YwMjZiLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MzgxN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA1ZjRjOTNhMGE5NjAwZDA4MzBlNzUwNjQyNjM1NGE4ZGI3MzQ0ZDAwOWY4NWEzYmU1ZmE1MmJiYjIyNTkwYTUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.5Koq-gbbYOX8XSyARRlJ5S7qJ7JZM8PPlJZQu5Nzskw)

# youtube apisinden çekilen metin
- Api ile çekilen metin ise şu şekilde temizlenip kayıt edilmelidir.Zira veri her biri zaman damgalı cümle topluluklarından oluşur.

![api metni](https://private-user-images.githubusercontent.com/69633060/294955046-623f4e00-5e22-44e6-b3dd-11318dc657d1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3Mzg1MTcsIm5iZiI6MTcwNDczODIxNywicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTUwNDYtNjIzZjRlMDAtNWUyMi00NGU2LWIzZGQtMTEzMThkYzY1N2QxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MjMzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTRlYzhlM2RjYmNjODVmMmYxNDRjYjM5MTEwODY3ZTRiZTBhZTdjMmVkYTY5NmQxOTRhYzZkMjgxOGYwZDVmOTImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.F6uGfgXVh1_rAe781oXa_IERHpyNUQ9CT1LRspG47eo)

## ham api metni:
![ham api metni](
https://private-user-images.githubusercontent.com/69633060/294955687-eef622b7-34b8-40d7-8b52-f1dc4b62b00a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ3MzgzNzcsIm5iZiI6MTcwNDczODA3NywicGF0aCI6Ii82OTYzMzA2MC8yOTQ5NTU2ODctZWVmNjIyYjctMzRiOC00MGQ3LThiNTItZjFkYzRiNjJiMDBhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA4VDE4MjExN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJjZTE1N2UzOTg4NTkxOTVlZjI3Y2Y1NjNkOWU4NTQ0YzAwNTJiNDc4MGQ5MWM0MDUxMzU5ZTFiMDU0NmE1MzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0._bXmpeZJEaX5p-cWaqOhc6GQjXN_7Y3ju3zPHesAWLo)


# Sonuç
Sonuçlar kullanılan veri setiyle beraber çok dalgalanmıştır.Borsa yorumları çok fazla spesifik görüş içerir ve kelime dağarcığı borsada aktif olan her kesimden insanın düşüncelerini ve cümle yapılarını içerir dolayısıyla konu hakkında tahmin yapmak oldukça zorlaşır.Burda farklı kelimeleri içeren yorumları bulma konusunda oldukça zayıf kaldım.19.000'e yakın yorum kullansam bile doğruluk oranlarını %84'e zor çıkardım burda çeşitli algoritmalarla beraber optimizasyon seçeneklerini de yerine getirdim.Cümleleri köklerini bulma işlemine sokmak onların olumsuz olma kısımlarını oldukça sınırlıyo bu anlamda cümledeki 'iyi', 'kötü' , 'güvenilmez' ,'alınmaz' vsvs. keliemeler dil bilgisi kuralları kullanılmaksızın pekte anlamlı bir kullanım içermiyo.Burda kullanıcıların iyi yorumlarında iyi gibi kelimeleri daha çok kullandığını düşünmek gerekiyor.Bu tip modellerde SVM çok faydalı il yapıyor, bir yapay sinir ağında demeler gerçekleştirdin fakat onları burda paylaşmıyorum.Ama svm'nin kullandığım modeller arasında en iyi performans göstereni olduğunu söyleyebilirim.Youtube'un api erişiinde tokenizasyon kısıtlaması işleri zorlaştırıyor, günlük yenilendiğiiçin bazı günlerde denemeleri kesmek zorunda kaldım.Diğer bir problem youtube videolarının trankriptlerinin inidirilebilmesi için Videonun sahinini 3.taraf yazılımlara izin vermesi gerekiyor.Bu da projenin ancak anlaşmalı kanallarla birlikte işe yarayacağını gösteriyor.Son karşılaşılan problem video'da kişilerin bir kaç hisse hakkında karşılaştırmalı konuşması burda  konular bir birinden keskin bir sınırla ayrılmalı bahsedilen hisse senetlerinin akıbeti orataya çıksın.Ben modelimde hakkında konuşulan kısmın tamamını alıyorum belki yüzlerce kelime konuşuluyo burda yorum kısının daha kısaltılması sonucları etkiliycektir.
