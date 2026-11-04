from turkish_legal_assistant.ingest import html_to_lines, parse_articles

# Mimics mevzuat.gov.tr's Word-exported HTML: wrapped lines, footnote spans, soft hyphens.
HTML = """<html><head><style>p { color: red }</style></head><body>
<p>DENEME KANUNU</p><p>Kanun Numarası : 9999</p>
<p>BİRİNCİ KISIM</p><p>Genel Kısım</p>
<p>BİRİNCİ BÖLÜM</p><p>Genel Hükümler</p>
<p>Amaç<span class=MsoFootnoteReference>[1]</span></p>
<p><b>Madde 1 - </b>Bu Kanunun amacı
deneme yapmaktır.</p>
<p>(Ek fıkra: 1/1/2020-1111/1 md.) İkinci fıkra.</p>
<p>İKİNCİ BÖLÜM</p><p>Tanımlar</p>
<p>Kavramlar</p>
<p><b>MADDE 2/A- </b>(1) Tanım&shy;lar şunlardır:</p>
<p>a) İşçi,</p>
<p><b>GEÇİCİ MADDE 1 – </b>Geçiş hükmü.</p>
<p>9999 SAYILI KANUNA EK VE DEĞİŞİKLİK GETİREN MEVZUATIN</p>
<p>Geçici Madde 1</p>
</body></html>"""


def parse(keep=None):
    return parse_articles(html_to_lines(HTML), "9999", "Deneme Kanunu", keep)


def test_splits_into_articles_and_stops_at_amendment_table():
    assert [a.article_number for a in parse()] == ["1", "2/A", "Geçici 1"]


def test_article_metadata():
    first, second, provisional = parse()
    assert first.title == "Amaç"
    assert first.section == "BİRİNCİ KISIM: Genel Kısım > BİRİNCİ BÖLÜM: Genel Hükümler"
    assert second.title == "Kavramlar"
    assert second.section == "BİRİNCİ KISIM: Genel Kısım > İKİNCİ BÖLÜM: Tanımlar"
    assert provisional.title == ""
    assert first.source_url.endswith("MevzuatNo=9999&MevzuatTur=1&MevzuatTertip=5")


def test_article_text_is_cleaned():
    first, second, _ = parse()
    assert first.text == "Bu Kanunun amacı deneme yapmaktır.\n(Ek fıkra: 1/1/2020-1111/1 md.) İkinci fıkra."
    assert second.text == "(1) Tanımlar şunlardır:\na) İşçi,"


def test_keep_range_filters_numbered_articles_only():
    assert [a.article_number for a in parse(keep=range(2, 3))] == ["2/A"]
