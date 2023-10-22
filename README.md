# Masterthesis2023

Dear Reader, this is the code created and utilized by me, Alexander Thomin, as part of my Master thesis "The Significance of the Anti-NATO Narrative in the Russian State Discourses of War"

It consists of the LinkScraper, TextScraper and DataProcessor scripts, which collected the data necesary for my research and also analyzed it.

The scrips can be used to retive data from the website of the kremlin (best access with active VPN: www.kremlin.ru).
It does scrape all articles published on the website in a ceirtain time range. 
I inlcuded the files that Linkscraper and Filescraper created so one can imedeatly try out the DataProcessor.

Getting the scripts running:
As these scripts were mostly a mean to an end and not an end in itself, I did not optimize it to be OS agnostic, many paths are hardcoded and would need to be changes and generally my criteria was that if it works for me that was good enought.
-unzip the Ukraine_all.zip
-pip version list at the end of this text
-Some of the dependecies of the packages I could not find on Windows but they worked on Linux (Ubuntu) and MacOS.
-dependecies not automatically installed via pip were ChromeDriver, a lxml parser library and some of the dependciey for Dostoyevsky, such as facebooks Fasttext. 

Using the scrips:
-While the timeframe is currently hardcoded in my LinkScraper script this timeframe can be altert with relative ease. Note that in this case parts of the DataProccesor also have to be adapted.
-finetuning the options hardcoded into the chrome driver might be handy to increase performance for your usecase
-select the analysis case by deleting the hashtag before the case you want to run and put a hashtag before the eother to disbale it. I originally intended to write a more elegant way to do this, but later thought it would be better to put this time into refining the thesis.

These srcipts were created as part of my master thesis, but most likely there wont be any further updates or support for it. 
Free to use, with further questions you can reach out to me under my firstname.lastname@gmail.com (see username).






























Package                Version
---------------------- ------------------
annotated-types        0.6.0
apturl                 0.5.2
attrs                  23.1.0
bcrypt                 3.2.0
beautifulsoup4         4.12.2
blinker                1.4
blis                   0.7.11
Brlapi                 0.8.3
bs4                    0.0.1
catalogue              2.0.10
certifi                2023.7.22
chardet                4.0.0
click                  8.0.3
cloudpathlib           0.16.0
colorama               0.4.4
command-not-found      0.3
confection             0.1.3
contourpy              1.1.1
cryptography           3.4.8
cupshelpers            1.0
cycler                 0.12.1
cymem                  2.0.8
DateTime               5.2
DAWG-Python            0.7.2
dbus-python            1.2.18
defer                  1.0.6
distro                 1.7.0
distro-info            1.1+ubuntu0.1
docopt-ng              0.9.0
dostoevsky             0.6.0
duplicity              0.8.21
exceptiongroup         1.1.3
fasteners              0.14.1
fasttext               0.9.2
fonttools              4.43.1
funcy                  2.0
future                 0.18.2
gensim                 4.3.2
h11                    0.14.0
httplib2               0.20.2
icu                    0.0.1
idna                   3.3
importlib-metadata     4.6.4
jeepney                0.7.1
Jinja2                 3.1.2
joblib                 1.3.2
keyring                23.5.0
kiwisolver             1.4.5
langcodes              3.3.0
language-selector      0.1
launchpadlib           1.10.16
lazr.restfulclient     0.14.4
lazr.uri               1.0.6
lockfile               0.12.2
louis                  3.20.0
macaroonbakery         1.3.1
Mako                   1.1.3
MarkupSafe             2.0.1
matplotlib             3.8.0
monotonic              1.6
more-itertools         8.10.0
Morfessor              2.0.6
murmurhash             1.0.10
netifaces              0.11.0
nltk                   3.8.1
numexpr                2.8.7
numpy                  1.26.1
oauthlib               3.2.0
olefile                0.46
outcome                1.3.0
packaging              23.2
pandas                 2.1.1
paramiko               2.9.3
pexpect                4.8.0
phik                   0.12.3
Pillow                 9.0.1
pip                    23.3
polyglot               16.7.4
preshed                3.0.9
protobuf               3.12.4
ptyprocess             0.7.0
pybind11               2.11.1
pycairo                1.20.1
pycld2                 0.41
pycups                 2.0.1
pydantic               2.4.2
pydantic_core          2.10.1
PyGObject              3.42.1
PyICU                  2.8.1
PyJWT                  2.3.0
pyLDAvis               3.4.1
pymacaroons            0.13.0
pymorphy3              1.2.1
pymorphy3-dicts-ru     2.4.417150.4580142
PyNaCl                 1.5.0
pyparsing              2.4.7
pyRFC3339              1.1
PySocks                1.7.1
python-apt             2.4.0+ubuntu2
python-dateutil        2.8.2
python-debian          0.1.43+ubuntu1.1
pytz                   2022.1
pyxdg                  0.27
PyYAML                 5.4.1
razdel                 0.5.0
regex                  2023.10.3
reportlab              3.6.8
requests               2.25.1
ru-core-news-lg        3.7.0
ru-core-news-md        3.7.0
ru-core-news-sm        3.7.0
scikit-learn           1.3.1
scipy                  1.11.3
SecretStorage          3.3.1
selenium               4.14.0
setuptools             68.2.2
six                    1.16.0
smart-open             6.4.0
sniffio                1.3.0
sortedcontainers       2.4.0
soupsieve              2.5
spacy                  3.7.2
spacy-legacy           3.0.12
spacy-loggers          1.0.5
spacytextblob          4.0.0
srsly                  2.4.8
systemd-python         234
textblob               0.15.3
thinc                  8.2.1
threadpoolctl          3.2.0
tqdm                   4.66.1
trio                   0.22.2
trio-websocket         0.11.1
typer                  0.9.0
typing_extensions      4.8.0
tzdata                 2023.3
ubuntu-advantage-tools 8001
ubuntu-drivers-common  0.0.0
ufw                    0.36.1
unattended-upgrades    0.1
urllib3                1.26.5
usb-creator            0.3.7
wadllib                1.3.6
wasabi                 1.1.2
weasel                 0.3.3
wheel                  0.37.1
wsproto                1.2.0
xdg                    5
xkit                   0.0.0
zipp                   1.0.0
zope.interface         6.1
