import sys; sys.path.insert(0,'/home/claude/vid'); from vo import build
V1 = [(4.5, [("Pakistan didn't conquer this city...", "Pakistan didn't conquer this city..."), ('it BOUGHT it.', 'it bought it.')]),
 (6.5, [('For 175 years, Gwadar was NOT part of Pakistan.', 'For a hundred and seventy-five years, Gwadar was not part of Pakistan.'), ('It belonged to the Sultan of Oman.', 'It belonged to the Sultan of Oman.')]),
 (6.0, [('This coastal strip was its own enclave...', 'This coastal strip was its own enclave...'), ('Omani from 1783 all the way to 1958.', 'Omani from seventeen eighty-three, all the way to nineteen fifty-eight.')]),
 (6.5, [('Then Pakistan paid around £3 million...', 'Then Pakistan paid around three million pounds...'), ('and by December 1958, Gwadar was Pakistani.', 'and by December, nineteen fifty-eight, Gwadar was Pakistani.')]),
 (7.5, [('Why does Gwadar look like this?', 'Why does Gwadar look like this?'), ('A rocky hammerhead, joined to land by sand...', 'A rocky hammerhead, joined to land by sand...'), ('creating two natural bays on either side.', 'creating two natural bays on either side.')]),
 (7.0, [('Today it is a deep-sea port...', 'Today, it is a deep-sea port...'), ('and the gateway of CPEC to China.', 'and the gateway of C-PEC to China.'), ('Follow for more!', 'Follow for more!')])]
V2 = [(4.5, [('Nearly 1,000 workers died...', 'Nearly one thousand workers died...'), ('building this one road.', 'building this one road.')]),
 (6.5, [('The Karakoram Highway...', 'The Karakoram Highway...'), ('1,300 km from Pakistan to Kashgar, China.', 'thirteen hundred kilometres, from Pakistan to Kashgar, China.')]),
 (7.0, [('It climbs to 4,693 metres at Khunjerab Pass...', 'It climbs to four thousand, six hundred and ninety-three metres, at Khunjerab Pass...'), ('the highest paved border crossing on Earth.', 'the highest paved border crossing on Earth.')]),
 (7.0, [('Pakistan first favoured the Mintaka Pass...', 'Pakistan first favoured the Mintaka Pass...'), ('but China chose Khunjerab — safer from air strikes.', 'but China chose Khunjerab, safer from air strikes.')]),
 (7.0, [('About 810 Pakistani workers lost their lives...', 'About eight hundred and ten Pakistani workers lost their lives...'), ('and about 200 Chinese — mostly in landslides and falls.', 'and about two hundred Chinese, mostly in landslides and falls.')]),
 (6.5, [("That's why they call it the 8th Wonder of the World.", "That's why they call it the eighth wonder of the world."), ('Follow for more!', 'Follow for more!')])]
build(V1, '/home/claude/vid/out_gwadar/vo.json', '/home/claude/vid/out_gwadar/vo.npy')
build(V2, '/home/claude/vid/out_kkh/vo.json', '/home/claude/vid/out_kkh/vo.npy')
