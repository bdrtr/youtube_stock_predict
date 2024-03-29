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
![data_set](image/assets/veri_seti_foto.png)


# Yorumların Temizlenmesi ve Lemmatization İşlemi
Veriseti oluşturulduktan sonra modelin daha iyi çalışması ve başarı oranının daha yüksek olması için yorumların temizlenmesi gerekmektedir. Yorumların içerisinde emojiler, noktalama işaretleri, stopwordsler, linkler gibi istenmeyen ve modelin başarısını düşürecek veriler yorumlar içerisinden temizleniyor. Daha sonra lemmatization (kelimelerin köklerinin alınması) işlemi yapılarak temiz ve kelimelerin köklerinden oluşan yorumlar elde ediliyor.

### Oluşturulan Temiz Yorum Görseli
![clean](image/assets/hisse_adi.png)


# Modelin Oluşturulması ve Yorumların Kategorilendirilmesi

## Model Seçimi
Yapılacak kategorilendirme işleminin hangi modelde daha yüksek başarı oranı vereceğini tespit etmek amacıyla araştırma yapılıp aynı zamanda bazı modeller üzerinde de test edilmiştir. Başlangıç olarak 3 popüler model üzerinde denemeler yapılmıştır. Bu modeller Naive Bayes, DecitionTree ve SVM modelidir. Veriseti üzerinde bu modellerin accuracy ve f1 score ları test edilmiştir. Projedeki test veriseti sonuçlarına bakıldığında:

- Naive Bayes Modeli için  f1 score: 0.710 
- DecitionTree Modeli için f1 score: 0.818
- SVM için (linear) f1 score: 0.765
- SVM (rbf) için f1 score: 0.807

![](image/assets/svm_analysis.png)

kullanılan dilin farklılığı ve teknik analiz kavramları doğruluk oranlarını oldukça düşürmektedir.


Sonuçlar incelendiğinde DecitionTree ve SVM (linear c=10) modelinin projede kullanılan verisetine göre vericeği sonuçların başarısı yeterlidir. Bu iki modelin kullanımına karar verilmiştir.

Support Vector Machine modeli kullanılmıştır. 

## Modelin Oluşturulmaya Başlanması

### Etiketleme
Yapılan yorumların bulunduğu "yorumlar.csv" dosyası dataframeye aktarılmıştır. Yorumların ifade ettiği değerlendirmeler -1 0 1 gibi bilgisayarın anlayabileceği bir formata dönüştürülmelidir. 'etiketli' adında yeni bir kolon açılarak olumsuz yorumlar için -1 sayısı, olumlu yorumlar için  1 sayısı, tarafsız yorumlar için 0 sayısı eklenmiştir.
 
![](image/assets/etiketleme.png)

### Verisetinin Parçalanması
Modelin başarısını doğru şekilde ölçebilmek için modeli eğittiğimiz veriler ile test ettiğimiz veriler farklı olmalıdır. Modelin eğitilmiş olduğu verileri tekrar modele gönderirsek model bu veriler ile eğitildiği için başarısı yüksek ve yanıltıcı olacaktır. Bu yüzden verisetini parçalamamız gerekir. Bu projede verisetinin %80 'i modeli eğitmek için, %20 'si de modeli test etmek için kullanılacaktır.

Verisetini parçalamak için train_test_split() fonksiyonu kullanılmıştır.

### Yorumlar Vektörel Matrisinin Çıkarılması
Yorumlar metinden oluştuğu için bunun bilgisayar ortamında işlenmesi mümkün değildir bu yüzden veriler sayısal değerlere dönüştürülmelidir. Bir sözlük oluşturularak dökümandaki her kelime için bir indexleme yapılır. Daha sonra hangi index numarasına sahip kelimenin hangi yorumda kaç kere geçtiği hesaplanarak sayma matrisi oluşturulur. Bu işlemi yaparken tf-idf vectorizer kullanılarak bir kelimenin döküman içindeki önemi istatistiksel olarak hesaplanmıştır. Bu sayede her yorumda geçen  anlamsız kelimelerin önemi düşürülmüştür yani stopwordsler tekrardan ayıklanmıştır.

### Modellerin Eğitilmesi
Daha önceden parçalanmış olan X_train ve y_train verileri Naive Bayes ve Support Vector Machine modeline gönderilerek modeller eğitilmiştir. Eğitim sonucunda modellerin accuracy ve f1 score değerleri hesaplanmıştır. Modelleri eğitmek için sklearn kütüphanesi kullanılmıştır.

### Modellerin Başarısının Hesaplanması
Modelin başarısı hem train hem test verileri üzerinden Accuracy ve F1 score ile ölçülmüştür. Alınan sonuçlar aşağıda bulunmaktadır.

Naive Bayes                |  Support Vector Machine  
:-------------------------:|:-------------------------: 
|![](image/assets/bayes_acc.png)| ![](image/assets/svm_acc.png)  |

### Modellerin Hata Oranları
- Modelin inceliklerini anlamaya çalıştığımızda yorumlar keskinleşince bayes yanlış sonuç veriyor.

![inceleme](image/assets/model_inceleme.png)

Modellerin hata oranlarını tespit etmek için 2 farklı metrik kullanılmıştır. Bunlar Ortalama Kare Hatası(MSE) ve Ortalama Mutlak Hata(MAE) dir.

##### Ortalama Kare Hatası(MSE)
Ortalama Kare Hatası tahmin edilen sonuçlarınızın gerçek sayıdan ne kadar farklı olduğuna dair size mutlak bir sayı verir.

##### Ortalama Mutlak Hata(MAE)
Ortalama mutlak hata, mutlak hata değerinin toplamını alır, hata terimlerinin toplamının daha doğrudan bir temsilidir.

![image](image/assets/msqe.png)

Sonuçlar incelendiğinde Support Vector Machine modelinin Naive Bayes modeline göre biraz daha az hata yaptığını görüyoruz.


### Manuel kontrol
Modeli manuel olarak test etmek için elle bazı yorumlar girilecek ve modelin bu yorumların hangi durumlara ait olduğunu tahmin etmesi istenecektir. Test sonuçları aşağıda gösterilmiştir.


##### Test1:
![test1](image/assets/test1.png)

##### Test2:
![test2](image/assets/test2.png)

##### Test3
![test3](image/assets/test3.png)

##### Test4
![test4](image/assets/test4.png)

##### Test5
![test5](image/assets/test5.png)

# youtube apisinden çekilen metin
- Api ile çekilen metin ise şu şekilde temizlenip kayıt edilmelidir.Zira veri her biri zaman damgalı cümle topluluklarından oluşur.

![api metni](image/assets/api_metni.png)

## ham api metni:
![ham api metni](
image/assets/ham_api_metni.png)


# Sonuç
Sonuçlar, kullanılan veri setiyle birlikte oldukça dalgalı olmuştur. Borsa yorumları çok spesifik görüş içerir ve kelime dağarcığı, borsada aktif olan her kesimden insanın düşüncelerini ve cümle yapılarını içerir. Dolayısıyla, konu hakkında tahmin yapmak oldukça zorlaşır. Burada farklı kelimeleri içeren yorumları bulma konusunda oldukça zayıf kaldım. 19.000'e yakın yorum kullansam bile doğruluk oranlarını %84'e çıkarmak zor oldu. Bu noktada çeşitli algoritmalar ve optimizasyon seçeneklerini kullanarak çalıştım. Cümleleri köklerine ayırmak, olumsuz ifadeleri sınırlıyor, bu da dil bilgisi kuralları kullanılmaksızın anlamlı bir çıkarım yapmayı güçleştiriyor. Kullanıcıların olumlu yorumlarında "iyi" gibi kelimelerin daha sık kullanıldığını düşünmek gerekiyor. Bu tip modellerde SVM'nin oldukça faydalı olduğunu gözlemledim. Bir yapay sinir ağında gerçekleştirdiğim denemeler de olumlu sonuçlar verdi. Ancak, bu sonuçları burada paylaşmıyorum. Youtube'un API erişiminde tokenizasyon kısıtlamaları işleri zorlaştırıyor ve günlük yenilendiği için bazı günler denemeleri kesmek zorunda kaldım. Diğer bir problem ise Youtube videolarının transkriptlerini indirebilmek için videonun sahibinin 3. taraf yazılımlara izin vermesi gerekiyor. Bu durum projenin sadece anlaşmalı kanallarla çalışabileceğini gösteriyor. Son karşılaşılan problem ise videolarda kişilerin birkaç hisse hakkında karşılaştırmalı konuşması. Burada konuların birbirinden keskin bir sınırla ayrılması gereklidir. Bahsedilen hisse senetlerinin akıbeti daha net bir şekilde ortaya çıkmalıdır. Ben modelimde hakkında konuşulan kısmın tamamını alıyorum, ancak bu durum yorum kısmının daha kısaltılmasıyla sonuçları etkileyebilir.
