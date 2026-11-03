from .models import Article, Law

# Sample articles for Phase 0. Phase 1 replaces these with texts scraped from mevzuat.gov.tr.
ARTICLES: list[Article] = [
    Article(
        law_number="4857",
        law_name="İş Kanunu",
        article_number="1",
        title="Amaç ve kapsam",
        text=(
            "Bu Kanunun amacı işverenler ile bir iş sözleşmesine dayanarak çalıştırılan "
            "işçilerin çalışma koşulları ve çalışma ortamına ilişkin hak ve sorumluluklarını "
            "düzenlemektir."
        ),
    ),
    Article(
        law_number="4857",
        law_name="İş Kanunu",
        article_number="53",
        title="Yıllık ücretli izin hakkı ve izin süreleri",
        text=(
            "İşyerinde işe başladığı günden itibaren, deneme süresi de içinde olmak üzere, "
            "en az bir yıl çalışmış olan işçilere yıllık ücretli izin verilir. İşçilere "
            "verilecek yıllık ücretli izin süresi, hizmet süresi; a) Bir yıldan beş yıla kadar "
            "(beş yıl dahil) olanlara ondört günden, b) Beş yıldan fazla onbeş yıldan az "
            "olanlara yirmi günden, c) Onbeş yıl (dahil) ve daha fazla olanlara yirmialtı "
            "günden az olamaz."
        ),
    ),
    Article(
        law_number="6698",
        law_name="Kişisel Verilerin Korunması Kanunu",
        article_number="1",
        title="Amaç",
        text=(
            "Bu Kanunun amacı, kişisel verilerin işlenmesinde başta özel hayatın gizliliği "
            "olmak üzere kişilerin temel hak ve özgürlüklerini korumak ve kişisel verileri "
            "işleyen gerçek ve tüzel kişilerin yükümlülükleri ile uyacakları usul ve esasları "
            "düzenlemektir."
        ),
    ),
    Article(
        law_number="6502",
        law_name="Tüketicinin Korunması Hakkında Kanun",
        article_number="1",
        title="Amaç",
        text=(
            "Bu Kanunun amacı; kamu yararına uygun olarak tüketicinin sağlık ve güvenliği ile "
            "ekonomik çıkarlarını koruyucu, zararlarını tazmin edici, çevresel tehlikelerden "
            "korunmasını sağlayıcı, tüketiciyi aydınlatıcı ve bilinçlendirici önlemleri almak, "
            "tüketicilerin kendilerini koruyucu girişimlerini özendirmek ve bu konulardaki "
            "politikaların oluşturulmasında gönüllü örgütlenmeleri teşvik etmeye ilişkin "
            "hususları düzenlemektir."
        ),
    ),
    Article(
        law_number="6098",
        law_name="Türk Borçlar Kanunu",
        article_number="299",
        title="Tanımı",
        text=(
            "Kira sözleşmesi, kiraya verenin bir şeyin kullanılmasını veya kullanılmasıyla "
            "birlikte ondan yararlanılmasını kiracıya bırakmayı, kiracının da buna karşılık "
            "kararlaştırılan kira bedelini ödemeyi üstlendiği sözleşmedir."
        ),
    ),
]

LAWS: list[Law] = list({a.law_number: Law(number=a.law_number, name=a.law_name) for a in ARTICLES}.values())
