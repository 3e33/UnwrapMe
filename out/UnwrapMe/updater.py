# Copyright (C) 2023-2026 Elias V.
# SPDX-License-Identifier: GPL-3.0-only

import urllib.request
import json
import os
import re
import zipfile
import shutil
import re
import platform

def getLatestReleaseURL(owner, repo):
    req = urllib.request.Request("https://api.github.com/repos/{}/{}/releases"
                                 .format(owner, repo))

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())

        searchString = ""
        if(platform.system() == "Windows"):
            searchString = "UnwrapMe.+?(\d.+?)-windows"

        if(platform.system() == "Linux"):
            searchString = "UnwrapMe.+?(\d.+?)-linux"

        if(not searchString):
            raise Exception("OS not detected or unsupported.")

        try:
            for a in data[0]["assets"]:
                if(re.search(searchString, a["browser_download_url"])):
                    return a["browser_download_url"]
        except:
            return None

def getTupleFromString(s):
    intTuple = tuple(map(int, re.findall(r'\d+', s)))

    return intTuple
