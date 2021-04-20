


knobs_str = """

"knobs": {
    "11": {
      "speedup": 1.0,
      "baseline_priority": 1,
      "devices": [
        "cpu"
      ]
    },
    "12": {
      "speedup": 1.5,
      "baseline_priority": 0,
      "devices": [
        "gpu"
      ]
    },
    "121": {
      "speedup": 2,
      "devices": [
        "cpu"
      ]
    },
    "122": {
      "speedup": 2,
      "devices": [
        "cpu"
      ]
    },
    "123": {
      "speedup": 2,
      "devices": [
        "cpu"
      ]
    },
    "124": {
      "speedup": 2,
      "devices": [
        "cpu"
      ]
    },
    "125": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "126": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "127": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "128": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "129": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "130": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "131": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "132": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "133": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "134": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "135": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "136": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "137": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "138": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "231": {
      "speedup": 2,
      "devices": [
        "cpu"
      ]
    },
    "232": {
      "speedup": 2,
      "devices": [
        "cpu"
      ]
    },
    "233": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "234": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "235": {
      "speedup": 1.5,
      "devices": [
        "cpu"
      ]
    },
    "236": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "237": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "238": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "239": {
      "speedup": 1.33,
      "devices": [
        "cpu"
      ]
    },
    "151": {
      "speedup": 3.0,
      "devices": [
        "gpu"
      ]
    },
    "152": {
      "speedup": 3.0,
      "devices": [
        "gpu"
      ]
    },
    "153": {
      "speedup": 3.0,
      "devices": [
        "gpu"
      ]
    },
    "154": {
      "speedup": 3.0,
      "devices": [
        "gpu"
      ]
    },
    "155": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "156": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "157": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "158": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "159": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "160": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "161": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "162": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "163": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "164": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "165": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "166": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "167": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "168": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "261": {
      "speedup": 3.0,
      "devices": [
        "gpu"
      ]
    },
    "262": {
      "speedup": 3.0,
      "devices": [
        "gpu"
      ]
    },
    "263": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "264": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "265": {
      "speedup": 2.25,
      "devices": [
        "gpu"
      ]
    },
    "266": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "267": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "268": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    },
    "269": {
      "speedup": 2.0,
      "devices": [
        "gpu"
      ]
    }
  },


"""


conv_knobs = "\"11\", \"12\","

for i in range(121, 139):
    conv_knobs += "\"" + str(i) + "\""
    conv_knobs += ", "

for i in range(151, 169):
    conv_knobs += "\"" + str(i) + "\""
    conv_knobs += ", "

for i in range(231, 240):
    conv_knobs += "\"" + str(i) + "\""
    conv_knobs += ", "

for i in range(261, 270):
    conv_knobs += "\"" + str(i) + "\""
    if i != 269:
        conv_knobs += ", "


baseline_knobs = "\"11\", \"12\""


