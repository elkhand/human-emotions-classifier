#Get OASIS dataset

wget --header 'Host: dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:59.0) Gecko/20100101 Firefox/59.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Cookie: uc_session=JyZlyPBRYUrnotOiBp5AFOwEHawL8sbJ33LRG7D1tmbPNsarHp9psp6wJ90KU6j4' --header 'Upgrade-Insecure-Requests: 1' 'https://dl.dropboxusercontent.com/content_link_zip/ILGO9iHS3gq83FkPwDj3AEdphC8x4M5680cGZMoxs6UFb4W8kDnE006KqYO9Uuj4/file?_download_id=34023872535984681820129128972114492684869972114547899610187031046&_notify_domain=www.dropbox.com&dl=1' --output-document 'images.zip'

unzip images.zip -d images
rm images.zip
