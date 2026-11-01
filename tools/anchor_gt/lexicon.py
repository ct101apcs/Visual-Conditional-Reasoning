"""Regex lexicon for anchor ground-truth checks on COCONut-PanCap captions.

Standard library only.  Provides category/synonym matching, head-chunk extraction,
group/partial/reflection/uncertain flags, subject typing, caption scene tags and
mask-based indoor/outdoor votes.
"""
import re
from functools import lru_cache

THING_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

PHYS_SIZE = {
    'person': 4, 'bicycle': 4, 'car': 5, 'motorcycle': 4, 'airplane': 6, 'bus': 6, 'train': 6,
    'truck': 6, 'boat': 5, 'traffic light': 3, 'fire hydrant': 3, 'stop sign': 3,
    'parking meter': 4, 'bench': 4, 'bird': 2, 'cat': 3, 'dog': 4, 'horse': 5, 'sheep': 4,
    'cow': 5, 'elephant': 6, 'bear': 5, 'zebra': 5, 'giraffe': 6, 'backpack': 3, 'umbrella': 3,
    'handbag': 3, 'tie': 2, 'suitcase': 3, 'frisbee': 2, 'skis': 4, 'snowboard': 4,
    'sports ball': 2, 'kite': 3, 'baseball bat': 3, 'baseball glove': 2, 'skateboard': 3,
    'surfboard': 4, 'tennis racket': 3, 'bottle': 2, 'wine glass': 2, 'cup': 2, 'fork': 1,
    'knife': 1, 'spoon': 1, 'bowl': 2, 'banana': 2, 'apple': 2, 'sandwich': 2, 'orange': 2,
    'broccoli': 2, 'carrot': 2, 'hot dog': 2, 'pizza': 3, 'donut': 2, 'cake': 3, 'chair': 4,
    'couch': 5, 'potted plant': 3, 'bed': 5, 'dining table': 4, 'toilet': 4, 'tv': 3,
    'laptop': 3, 'mouse': 1, 'remote': 2, 'keyboard': 3, 'cell phone': 1, 'microwave': 3,
    'oven': 4, 'toaster': 3, 'sink': 4, 'refrigerator': 4, 'book': 2, 'clock': 3, 'vase': 3,
    'scissors': 2, 'teddy bear': 3, 'hair drier': 2, 'toothbrush': 1}

# ----------------------------------------------------------------------------
# Synonym table: category -> singular terms (a term may belong to several categories)
# ----------------------------------------------------------------------------
_SPORTS = ('baseball|tennis|soccer|football|basketball|golf|volleyball|rugby|cricket|hockey|'
           'polo|frisbee|lacrosse|softball|badminton|squash|handball|ping pong|table tennis')
_PERSON_SPORT = ['player', 'batter', 'pitcher', 'catcher', 'fielder', 'umpire', 'referee', 'coach',
                 'goalie', 'goalkeeper', 'fan', 'competitor']

SYNONYMS = {
    'person': [
        'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'infant', 'toddler',
        'player', 'skier', 'snowboarder', 'surfer', 'skateboarder', 'rider', 'lady', 'guy',
        'gentleman', 'chef', 'cook', 'officer', 'batter', 'catcher', 'pitcher', 'umpire',
        'referee', 'individual', 'figure', 'spectator', 'pedestrian', 'people', 'human', 'adult',
        'teenager', 'teen', 'youngster', 'swimmer', 'runner', 'skater', 'cyclist', 'biker',
        'motorcyclist', 'driver', 'passenger', 'tourist', 'worker', 'customer', 'vendor', 'waiter',
        'waitress', 'bride', 'groom', 'student', 'athlete', 'fielder', 'outfielder', 'infielder',
        'shortstop', 'baseman', 'goalkeeper', 'goalie', 'coach', 'wakeboarder', 'kiteboarder',
        'paraglider', 'kayaker', 'sailor', 'fisherman', 'farmer', 'soldier', 'policeman',
        'policewoman', 'cop', 'mother', 'father', 'mom', 'dad', 'parent', 'dancer', 'musician',
        'singer', 'performer', 'photographer', 'mailman', 'jockey', 'bystander', 'onlooker',
        'traveler', 'traveller', 'hiker', 'climber', 'pilot', 'someone', 'somebody', 'silhouette',
        'crowd', 'audience', 'folks', 'persons', 'gal', 'fella', 'dude', 'boyfriend', 'girlfriend',
        'husband', 'wife', 'spectators', 'snowboarders', 'skiers', 'toddlers', 'surfers',
        'pedestrians', 'cowboy', 'cowgirl', 'skiier', 'batsman', 'bowler', 'wicketkeeper',
        'goalkeepers', 'lifeguard', 'bartender', 'barista', 'nurse', 'doctor', 'surgeon',
        'shopper', 'commuter', 'cyclists', 'men', 'women', 'children', 'ladies', 'gentlemen',
        'kids', 'babies', 'catchers', 'pitchers',
    ],
    'bicycle': [
        'bicycle', 'bike', 'mountain bike', 'bmx', 'tandem bicycle', 'road bike', 'tricycle',
        'unicycle', 'cruiser bike', 'bicycles', 'bikes',
    ],
    'car': [
        'car', 'sedan', 'suv', 'taxi', 'cab', 'taxi cab', 'taxicab', 'automobile', 'hatchback',
        'sports car', 'police car', 'minivan', 'van', 'jeep', 'convertible', 'coupe', 'limousine',
        'limo', 'station wagon', 'pickup', 'cars',
    ],
    'motorcycle': [
        'motorcycle', 'motorbike', 'scooter', 'moped', 'dirt bike', 'dirtbike', 'motor bike',
        'motor cycle', 'motor scooter', 'vespa', 'motorcycles', 'chopper',
    ],
    'airplane': [
        'airplane', 'aeroplane', 'plane', 'jet', 'aircraft', 'airliner', 'jetliner', 'fighter jet',
        'helicopter', 'biplane', 'propeller plane', 'glider', 'seaplane', 'jumbo jet',
        'passenger plane', 'passenger jet', 'cargo plane', 'copter',
    ],
    'bus': [
        'bus', 'school bus', 'minibus', 'double decker', 'double decker bus', 'tour bus',
        'city bus', 'coach bus', 'shuttle', 'trolley bus', 'trolleybus', 'omnibus',
    ],
    'train': [
        'train', 'locomotive', 'train car', 'railcar', 'rail car', 'subway', 'tram', 'streetcar',
        'passenger train', 'railway car', 'boxcar', 'caboose', 'light rail', 'freight train',
        'steam engine', 'subway car',
    ],
    'truck': [
        'truck', 'pickup truck', 'lorry', 'semi', 'semi truck', 'dump truck', 'fire truck',
        'firetruck', 'garbage truck', 'tractor trailer', 'van', 'delivery truck', 'tow truck',
        'cement truck', 'pickup', 'big rig', 'tractor-trailer', 'semi trailer', 'fire engine',
    ],
    'boat': [
        'boat', 'ship', 'sailboat', 'canoe', 'kayak', 'yacht', 'ferry', 'fishing boat',
        'speedboat', 'rowboat', 'dinghy', 'vessel', 'paddle boat', 'catamaran', 'raft', 'barge',
        'tugboat', 'cruise ship', 'gondola', 'jet ski', 'jetski', 'motorboat', 'steamboat',
        'schooner', 'lifeboat', 'watercraft', 'sailing ship', 'sail boat',
    ],
    'traffic light': [
        'traffic light', 'traffic signal', 'stoplight', 'stop light', 'traffic lamp',
        'signal light', 'red light', 'green light', 'street signal',
    ],
    'fire hydrant': [
        'fire hydrant', 'hydrant', 'fire plug', 'fireplug',
    ],
    'stop sign': [
        'stop sign', 'stopsign',
    ],
    'parking meter': [
        'parking meter', 'meter', 'parking kiosk', 'pay station',
    ],
    'bench': [
        'bench', 'pew', 'park bench', 'picnic bench',
    ],
    'bird': [
        'bird', 'seagull', 'gull', 'pigeon', 'duck', 'goose', 'swan', 'parrot', 'owl', 'eagle',
        'hawk', 'crow', 'sparrow', 'penguin', 'ostrich', 'chicken', 'rooster', 'hen', 'pelican',
        'heron', 'flamingo', 'seabird', 'peacock', 'falcon', 'dove', 'robin', 'cockatoo', 'macaw',
        'hummingbird', 'finch', 'stork', 'duckling', 'raven', 'albatross', 'kingfisher',
        'parakeet', 'cormorant', 'egret', 'bluebird', 'woodpecker', 'sandpiper', 'puffin',
        'toucan', 'vulture', 'goslings', 'gosling', 'waterfowl', 'shorebird', 'songbird', 'avian',
    ],
    'cat': [
        'cat', 'kitten', 'kitty', 'feline', 'tabby', 'kittens', 'tomcat', 'housecat',
    ],
    'dog': [
        'dog', 'puppy', 'pup', 'canine', 'pooch', 'terrier', 'retriever', 'labrador', 'bulldog',
        'poodle', 'golden retriever', 'german shepherd', 'husky', 'beagle', 'dachshund',
        'chihuahua', 'corgi', 'pug', 'collie', 'spaniel', 'dalmatian', 'doggy', 'doggie',
        'greyhound', 'rottweiler', 'schnauzer', 'mutt', 'hound', 'sheepdog', 'whippet', 'maltese',
        'labradoodle', 'pitbull', 'pit bull', 'australian shepherd', 'border collie', 'great dane',
        'shih tzu', 'pomeranian', 'sled dog',
    ],
    'horse': [
        'horse', 'pony', 'stallion', 'mare', 'foal', 'colt', 'bronco', 'steed', 'racehorse',
        'clydesdale', 'mustang', 'draft horse',
    ],
    'sheep': [
        'sheep', 'lamb', 'ram', 'ewe', 'lambs', 'flock of sheep', 'sheeps',
    ],
    'cow': [
        'cow', 'cattle', 'bull', 'calf', 'ox', 'steer', 'bison', 'buffalo', 'heifer', 'bovine',
        'oxen', 'water buffalo',
    ],
    'elephant': [
        'elephant', 'calf elephant', 'tusker',
    ],
    'bear': [
        'bear', 'polar bear', 'grizzly', 'grizzly bear', 'brown bear', 'black bear', 'panda',
        'panda bear', 'bear cub',
    ],
    'zebra': [
        'zebra', 'zebras',
    ],
    'giraffe': [
        'giraffe', 'giraffes',
    ],
    'backpack': [
        'backpack', 'rucksack', 'knapsack', 'back pack', 'daypack', 'bag', 'book bag', 'schoolbag',
        'school bag', 'bookbag',
    ],
    'umbrella': [
        'umbrella', 'parasol', 'sunshade', 'beach umbrella', 'canopy umbrella',
    ],
    'handbag': [
        'handbag', 'purse', 'bag', 'tote', 'tote bag', 'satchel', 'shoulder bag', 'messenger bag',
        'hand bag',
    ],
    'tie': [
        'tie', 'necktie', 'neck tie', 'bow tie', 'bowtie', 'cravat',
    ],
    'suitcase': [
        'suitcase', 'luggage', 'baggage', 'bag', 'carry on', 'carry-on', 'travel bag',
        'duffel bag', 'duffle bag', 'duffel', 'duffle', 'valise', 'rolling bag', 'baggage cart',
        'suit case', 'travel case',
    ],
    'frisbee': [
        'frisbee', 'flying disc', 'flying saucer', 'frisbees', 'flying disk',
    ],
    'skis': [
        'ski', 'skis', 'cross country skis',
    ],
    'snowboard': [
        'snowboard', 'snow board', 'snowboards',
    ],
    'sports ball': [
        'ball', 'sports ball', 'soccer ball', 'tennis ball', 'baseball', 'basketball', 'football',
        'volleyball', 'golf ball', 'beach ball', 'rugby ball', 'cricket ball', 'softball',
        'volley ball', 'foot ball', 'base ball', 'basket ball', 'dodgeball', 'kickball',
        'bowling ball', 'ping pong ball',
    ],
    'kite': [
        'kite', 'kites', 'kite surfing kite', 'parafoil',
    ],
    'baseball bat': [
        'baseball bat', 'bat', 'cricket bat', 'softball bat', 'wooden bat', 'base ball bat',
    ],
    'baseball glove': [
        'baseball glove', 'glove', 'mitt', 'baseball mitt', 'catchers mitt', 'fielders glove',
        'softball glove', 'ball glove', 'catcher glove',
    ],
    'skateboard': [
        'skateboard', 'skate board', 'longboard', 'skateboards', 'long board',
    ],
    'surfboard': [
        'surfboard', 'surf board', 'paddleboard', 'paddle board', 'wakeboard', 'boogie board',
        'body board', 'bodyboard', 'windsurf board', 'surfboards', 'kiteboard', 'kite board',
    ],
    'tennis racket': [
        'tennis racket', 'racket', 'racquet', 'tennis racquet', 'badminton racket',
        'squash racket', 'rackets', 'racquets',
    ],
    'bottle': [
        'bottle', 'water bottle', 'beer bottle', 'wine bottle', 'soda bottle', 'bottles',
        'sauce bottle', 'baby bottle', 'milk bottle', 'ketchup bottle', 'plastic bottle',
        'glass bottle',
    ],
    'wine glass': [
        'wine glass', 'wineglass', 'wine goblet', 'goblet', 'champagne flute', 'flute',
        'champagne glass', 'glass', 'stemware', 'martini glass', 'cocktail glass', 'wine glasses',
        'drinking glass', 'beer glass', 'tumbler', 'stemmed glass',
    ],
    'cup': [
        'cup', 'mug', 'coffee cup', 'teacup', 'tea cup', 'glass', 'paper cup', 'tumbler',
        'coffee mug', 'styrofoam cup', 'drinking glass', 'plastic cup', 'beer mug', 'cups',
        'cupful', 'sippy cup', 'travel mug', 'beer glass', 'espresso cup',
    ],
    'fork': [
        'fork', 'forks',
    ],
    'knife': [
        'knife', 'butcher knife', 'cleaver', 'pocket knife', 'kitchen knife', 'steak knife',
        'butter knife', 'bread knife', 'penknife', 'knives',
    ],
    'spoon': [
        'spoon', 'ladle', 'spatula', 'teaspoon', 'tablespoon', 'soup spoon', 'wooden spoon',
    ],
    'bowl': [
        'bowl', 'plate', 'platter', 'salad bowl', 'soup bowl', 'dish', 'cereal bowl',
        'mixing bowl', 'fruit bowl', 'serving bowl',
    ],
    'banana': [
        'banana', 'bananas',
    ],
    'apple': [
        'apple', 'apples', 'green apple', 'red apple',
    ],
    'sandwich': [
        'sandwich', 'burger', 'hamburger', 'cheeseburger', 'sub', 'panini', 'wrap',
        'submarine sandwich', 'club sandwich', 'sub sandwich', 'hoagie', 'sliders', 'slider',
        'sandwiches', 'grilled cheese',
    ],
    'orange': [
        'orange', 'tangerine', 'clementine', 'mandarin', 'tangelo', 'oranges',
    ],
    'broccoli': [
        'broccoli', 'broccolini',
    ],
    'carrot': [
        'carrot', 'carrots',
    ],
    'hot dog': [
        'hot dog', 'hotdog', 'corn dog', 'frankfurter', 'bratwurst', 'chili dog', 'sausage',
        'wiener', 'weiner', 'hot dogs', 'hot-dog',
    ],
    'pizza': [
        'pizza', 'pizzas', 'pizza pie', 'pizza slice',
    ],
    'donut': [
        'donut', 'doughnut', 'donuts', 'doughnuts', 'pastry', 'pastries', 'cruller',
    ],
    'cake': [
        'cake', 'cupcake', 'cheesecake', 'birthday cake', 'pastry', 'muffin', 'brownie',
        'cupcakes', 'dessert', 'torte', 'pastries', 'cakes', 'desserts', 'wedding cake',
        'layer cake', 'sheet cake', 'eclair', 'cheese cake', 'pastry slice',
    ],
    'chair': [
        'chair', 'armchair', 'stool', 'recliner', 'highchair', 'high chair', 'rocking chair',
        'office chair', 'lawn chair', 'wheelchair', 'wheel chair', 'barstool', 'bar stool',
        'deck chair', 'folding chair', 'chairs', 'beanbag', 'swivel chair', 'desk chair',
        'dining chair', 'patio chair', 'recliner chair', 'adirondack chair',
    ],
    'couch': [
        'couch', 'sofa', 'loveseat', 'love seat', 'futon', 'settee', 'sectional', 'divan',
        'couches', 'sofas', 'chesterfield',
    ],
    'potted plant': [
        'potted plant', 'plant', 'houseplant', 'house plant', 'flower pot', 'flowerpot', 'planter',
        'potted flower', 'potted tree', 'bonsai', 'succulent', 'cactus', 'fern', 'plant pot',
        'flower', 'flowers', 'bouquet', 'plants', 'floral arrangement',
    ],
    'bed': [
        'bed', 'bunk bed', 'crib', 'cot', 'mattress', 'bed frame', 'beds', 'bunkbed',
        'four poster bed', 'canopy bed',
    ],
    'dining table': [
        'dining table', 'table', 'desk', 'tabletop', 'table top', 'coffee table', 'picnic table',
        'kitchen table', 'end table', 'side table', 'work desk', 'tables', 'desks', 'countertop',
        'counter', 'console table', 'dinner table', 'island', 'kitchen island', 'table cloth',
        'tablecloth', 'work table', 'worktable', 'table surface', 'counter top',
    ],
    'toilet': [
        'toilet', 'commode', 'toilet bowl', 'urinal', 'toilets', 'lavatory', 'potty', 'wc',
        'toilet seat', 'bidet', 'toilet tank',
    ],
    'tv': [
        'tv', 'television', 'monitor', 'screen', 'tv screen', 'flat screen', 'flat-screen',
        'flatscreen', 'tv set', 'computer monitor', 'televisions', 'tvs', 'monitors', 'screens',
        'television set', 'projector screen', 'computer screen', 'flat screen tv', 'big screen',
        'widescreen', 'projection screen', 'display screen',
    ],
    'laptop': [
        'laptop', 'laptop computer', 'notebook computer', 'notebook', 'macbook', 'netbook',
        'computer', 'laptops', 'chromebook', 'ultrabook', 'notebook pc',
    ],
    'mouse': [
        'mouse', 'computer mouse', 'mice', 'wireless mouse', 'optical mouse',
    ],
    'remote': [
        'remote', 'remote control', 'controller', 'game controller', 'wii remote', 'wiimote',
        'remote controller', 'wii controller', 'remotes', 'nintendo wii controller', 'tv remote',
        'wii mote', 'console controller', 'controllers', 'remote controls',
    ],
    'keyboard': [
        'keyboard', 'computer keyboard', 'keyboards', 'keypad', 'laptop keyboard',
    ],
    'cell phone': [
        'cell phone', 'cellphone', 'phone', 'smartphone', 'mobile phone', 'iphone', 'telephone',
        'handset', 'phones', 'smart phone', 'android phone', 'flip phone', 'cellular phone',
        'blackberry',
    ],
    'microwave': [
        'microwave', 'microwave oven', 'microwaves',
    ],
    'oven': [
        'oven', 'stove', 'stovetop', 'cooktop', 'toaster oven', 'stoves', 'ovens', 'range top',
        'gas stove', 'stove top', 'cooking range', 'oven door', 'stove oven', 'electric stove',
        'cooking stove',
    ],
    'toaster': [
        'toaster', 'toaster oven', 'toasters',
    ],
    'sink': [
        'sink', 'basin', 'washbasin', 'wash basin', 'sinks', 'washbowl', 'bathroom sink',
        'kitchen sink', 'lavabo', 'washstand', 'basins',
    ],
    'refrigerator': [
        'refrigerator', 'fridge', 'freezer', 'refrigerators', 'fridges', 'icebox', 'mini fridge',
        'minifridge',
    ],
    'book': [
        'book', 'novel', 'notebook', 'textbook', 'books', 'magazine', 'magazines', 'paperback',
        'hardcover', 'notepad', 'journal', 'diary', 'bible', 'dictionary', 'encyclopedia', 'comic',
        'comic book', 'cookbook', 'photo album', 'sketchbook', 'scrapbook', 'notebooks',
    ],
    'clock': [
        'clock', 'clock tower', 'wall clock', 'alarm clock', 'clocks', 'clock face', 'timepiece',
        'grandfather clock', 'clocktower', 'clock tower face', 'digital clock', 'analog clock',
        'town clock', 'street clock',
    ],
    'vase': [
        'vase', 'pitcher', 'flowers', 'flower', 'bouquet', 'floral arrangement', 'flower vase',
        'urn', 'vases', 'bud vase', 'flower vases', 'planter vase',
    ],
    'scissors': [
        'scissors', 'shears', 'scissor', 'pair of scissors', 'shear',
    ],
    'teddy bear': [
        'teddy bear', 'teddy', 'stuffed animal', 'plush toy', 'stuffed toy', 'stuffed bear',
        'plush', 'bear', 'teddy bears', 'plush animal', 'stuffed bunny', 'plushie', 'toy bear',
        'bears', 'stuffed rabbit', 'stuffed dog', 'stuffed cat', 'stuffed elephant',
        'stuffed giraffe', 'stuffed lion', 'stuffed monkey', 'stuffed panda', 'plush bear',
        'cuddly toy', 'soft toy', 'plush doll',
    ],
    'hair drier': [
        'hair drier', 'hair dryer', 'hairdryer', 'blow dryer', 'blow drier', 'hairdrier',
        'blowdryer', 'dryer', 'drier', 'hair straightener', 'curling iron', 'flat iron',
        'hair iron', 'blow-dryer', 'hair-dryer',
    ],
    'toothbrush': [
        'toothbrush', 'tooth brush', 'toothbrushes', 'electric toothbrush',
    ],
}
# 'bag' is shared by handbag/backpack/suitcase, 'bear' by bear/teddy bear, 'pitcher' by person/vase, etc.

# Inherently plural / collective surface forms: is_plural=True for the form itself.
_INHERENT_PLURAL = {'people', 'cattle', 'folks', 'police'}
_INVARIANT = {'sheep', 'deer', 'fish', 'luggage', 'baggage', 'bison', 'cattle', 'people'}

_IRREGULAR_PLURAL = {'person': ['people', 'persons'], 'man': ['men'], 'woman': ['women'],
                     'child': ['children'], 'mouse': ['mice'], 'goose': ['geese'], 'ox': ['oxen'],
                     'knife': ['knives'], 'leaf': ['leaves'], 'calf': ['calves'], 'wife': ['wives'],
                     'shelf': ['shelves'], 'loaf': ['loaves'], 'foot': ['feet'], 'tooth': ['teeth'],
                     'fish': ['fish'], 'sheep': ['sheep'], 'deer': ['deer'], 'wolf': ['wolves'],
                     'half': ['halves'], 'potato': ['potatoes'], 'tomato': ['tomatoes'],
                     'baby': ['babies'], 'lady': ['ladies'], 'cattle': ['cattle'],
                     'bison': ['bison'], 'skis': ['skis'], 'ski': ['skis'], 'scissors': ['scissors'],
                     'shears': ['shears'], 'police': ['police'], 'people': ['people'],
                     'luggage': [], 'baggage': [], 'silverware': [], 'flatware': [], 'cutlery': [],
                     'foliage': [], 'greenery': [], 'vegetation': [], 'stemware': [], 'pottery': [],
                     'ceramic': [], 'seating': [], 'television': ['televisions'], 'bedding': [],
                     'equipment': [], 'crowd': ['crowds'], 'audience': ['audiences'],
                     'folks': [], 'oxen': [], 'mice': [], 'men': [], 'women': [], 'children': [],
                     'geese': [], 'knives': [], 'ladies': [], 'kids': [], 'babies': []}


def _pluralize(word):
    if word in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[word]
    if word.endswith('woman'):
        return [word[:-5] + 'women']
    if word.endswith('man') and word not in ('human', 'german', 'roman', 'caiman'):
        return [word[:-3] + 'men']
    if re.search(r'(?:s|x|z|ch|sh)$', word):
        return [word + 'es']
    if re.search(r'[^aeiou]y$', word):
        return [word[:-1] + 'ies']
    if word.endswith('fe'):
        return [word[:-2] + 'ves']
    return [word + 's']


def _norm(s):
    return re.sub(r'[\s\-]+', ' ', s.lower().strip())


# form (normalised) -> {category: is_plural}
_FORMS = {}


def _add_form(form, cat, plural):
    _FORMS.setdefault(form, {}).setdefault(cat, plural)


def _build_forms():
    terms = {c: list(dict.fromkeys(_norm(t) for t in SYNONYMS[c])) for c in THING_CATEGORIES}
    for pre in _SPORTS.split('|'):
        for suf in _PERSON_SPORT:
            terms['person'].append(pre + ' ' + suf)
    for cat, tl in terms.items():
        gen = set()
        for t in tl:
            words = t.split(' ')
            for p in _pluralize(words[-1]):
                gen.add(' '.join(words[:-1] + [p]))
        for t in tl:
            words = t.split(' ')
            if t in _INHERENT_PLURAL or (t in gen and t not in _INVARIANT):
                _add_form(t, cat, True)
                continue
            _add_form(t, cat, False)
            for p in _pluralize(words[-1]):
                pf = ' '.join(words[:-1] + [p])
                if pf != t:
                    _add_form(pf, cat, True)


_build_forms()

# Null terms: phrases that contain a category word but do not denote that category.
_NULL_PATTERNS = [
    r'(?:%s)[\s-]+(?:cap|caps|uniform|uniforms|field|game|games|match|court|stadium|jersey|jerseys|team|teams|'
    r'helmet|shoes?|outfit|skirt|dress|pitch|diamond|stands?|arena|net|pants|shirt|camp|practice|league|'
    r'crowd|course|club|season|park|socks?|gear|attire|equipment|bag|tournament|fans?|training|'
    r'helmets|cleats?|jacket|shorts|clothes|clothing|ground|grounds|event|action|scene|setting|'
    r'sneakers?|logo|poster|trophy|memorabilia|scoreboard|dugout|stadium|bench|bleachers)' % _SPORTS,
    r'ski[\s-]+(?:poles?|boots?|slopes?|resort|lift|jackets?|goggles?|gear|suits?|mask|hats?|helmets?|'
    r'trails?|runs?|area|lodge|gloves?|pants|equipment|outfit|hill|tracks?|patrol|school|instructor|'
    r'masks?|clothing|clothes|attire|slope|chalet|village|town|season|trip|vacation|bib|lifts)',
    r'snow[\s-]*board(?:ing)?[\s-]+(?:boots?|gear|jacket|helmet|goggles?|park|pants|gloves?|outfit|suit|'
    r'equipment|clothing|attire|slope)',
    r'(?:fried|roast|roasted|grilled|baked|bbq|barbecue|barbecued|crispy|breaded|rotisserie|buffalo) chicken',
    r'chicken[\s-]+(?:wings?|nuggets?|breasts?|salad|strips?|tenders?|legs?|thighs?|curry|soup|pot pie|parm)',
    r'(?:ice|whipped|sour) cream',
    r'(?:hot|cold|iced|coffee|tea|water) (?:chocolate|coffee|tea|water)',
    r'tie[\s-]+dye[d]?',
    r'man[\s-]made',
    r'glass[\s-]*(?:topped|covered|enclosed|walled|paneled|fronted)',
    r'(?:dining|living|bed|bath|class|board|conference|meeting|waiting|rest|show|store|work|dressing|'
    r'locker|changing|laundry|utility|control|engine|powder|tea|ball|dance|ware)[\s-]*room',
    r'(?:screen|mouse|computer|remote|cell|stuffed|bear)[\s-]+(?:saver|pad|cable|cord|market|shop|country)',
    r'mobile home',
    r'bus[\s-]+(?:stop|station|shelter|lane|driver|terminal|depot|stand)',
    r'(?:train|railway|subway|metro|tram)[\s-]+(?:station|stop|platform|tracks?|track|yard|depot|terminal|crossing|line|bridge|car park)',
    r'(?:taxi|car|cab|truck|plane|airplane|air|bus|bike|bicycle|boat)[\s-]+(?:stand|park|parking|lot|wash|rental|dealership|show|ride|rides|seat|door|wheel|window|hood|trunk|accident|crash|race|garage|lane|port|dock|ramp|lift|repair)',
    r'(?:fire|police|ambulance|rescue) (?:station|department)',
    r'(?:table|chair|desk|bed|couch|sofa|bench|clock|bowl|plant|pot|toilet|sink|oven|stove)[\s-]+(?:lamp|leg|legs|cloth|runner|setting|settings|top|tennis|decor|arrangement|mat|mats|saw|spoon|fan|skirt|cover|covers)',
    r'orange[\s-]+(?:colou?r(?:ed)?|and|&|or|shirts?|t[\s-]?shirts?|jackets?|vests?|tops?|hats?|caps?|sweaters?|'
    r'sweatshirts?|hoodies?|dress(?:es)?|shorts|pants|skirts?|helmets?|uniforms?|jerseys?|scarfs?|scarves|coats?|'
    r'aprons?|shoes|sneakers|gloves?|stripes?|striped|lines?|cones?|safety|glow|hue|tint|tone|tones|'
    r'sunset|sunrise|sky|light|lights|glowing|flames?|fire|paint|painted|tape|plastic|leaves|walls?|wall|'
    r'color|colors|colour|colours|sails?|curtains?|rope|jumpsuit|robe|robes|tie|ties|pillows?|cushions?|'
    r'blanket|towels?|ribbon|flag|flags|buoys?|barrels?|barriers?|barricades?|signs?|and white|black|white|blue|red|green|yellow|brown|gray|grey|pink|purple|tan|beige)',
    r'remote[\s-]+(?:location|area|village|island|place|setting|region|field|control(?:led)?\s*car|work|working|worker)',
    r'glass[\s-]+(?:doors?|windows?|buildings?|walls?|cases?|panels?|panes?|facades?|ceilings?|roofs?|'
    r'storefronts?|partitions?|shelf|shelves|displays?|enclosures?|bowls?|containers?|top|tops|'
    r'table|tables|tabletops?|plates?|dish(?:es)?|block|blocks|bricks?|sculptures?|art|fronts?|elevators?|'
    r'balcony|railings?|structure|skyscrapers?|dome|tiles?|mosaic|'
    r'lids?|lid|surface|slab|sheets?|paneled|paned|covered|wall|front|stairs|staircase|bridge|barrier|barriers|screens?)',
    r'(?:plant|pot)[\s-]+(?:based|stand|holder)',
    r'power plant',
    r'bat[\s-]*(?:girl|boy)',
    r'(?:pool|billiard|snooker)[\s-]+(?:table|ball|balls)',
    r'(?:cell)[\s-]+(?:tower|towers|phone tower)',
    r'(?:table|picnic)[\s-]+(?:umbrella|umbrellas)',
]

def _term_regex(form):
    return r'[\s-]+'.join(re.escape(w) for w in form.split(' '))


_NULL_RE = re.compile(r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % '|'.join('(?:%s)' % p for p in _NULL_PATTERNS), re.I)
_TOKEN_RE = re.compile(r'[A-Za-z]+')
_SEP_RE = re.compile(r'[\s-]+')
_MAXN = max(len(f.split(' ')) for f in _FORMS)

CATEGORY_PATTERNS = {}
for _cat in THING_CATEGORIES:
    _fs = sorted((f for f, d in _FORMS.items() if _cat in d), key=lambda f: (-len(f), f))
    CATEGORY_PATTERNS[_cat] = re.compile(
        r'(?<![A-Za-z])(?:%s)(?![A-Za-z])' % '|'.join(_term_regex(f) for f in _fs), re.I)

_COLOR_BEFORE = re.compile(
    r'(?<![A-Za-z])(?:red|white|blue|green|yellow|black|brown|gray|grey|pink|purple|tan|beige|silver|gold)'
    r'\s*(?:,|and|&|-|,\s*and|or)\s*$', re.I)


@lru_cache(maxsize=None)
def _categories_in(text):
    nulls = [(m.start(), m.end()) for m in _NULL_RE.finditer(text)]
    toks = [(m.group().lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    matches = []
    i = 0
    while i < len(toks):
        s0 = toks[i][1]
        if any(a <= s0 < b for a, b in nulls):
            i += 1
            continue
        hit = None
        for n in range(min(_MAXN, len(toks) - i), 0, -1):
            ok = True
            for k in range(i, i + n - 1):
                if not _SEP_RE.fullmatch(text[toks[k][2]:toks[k + 1][1]]):
                    ok = False
                    break
            if not ok:
                continue
            form = ' '.join(t[0] for t in toks[i:i + n])
            if form in _FORMS:
                hit = (n, form)
                break
        if hit:
            n, form = hit
            matches.append((s0, toks[i + n - 1][2], form, _FORMS[form]))
            i += n
        else:
            i += 1
    out = []
    for j, (s, e, form, d) in enumerate(matches):
        if form in ('orange', 'oranges'):
            if _COLOR_BEFORE.search(text[max(0, s - 20):s]):
                continue
            if j + 1 < len(matches):
                ns, _, _, nd = matches[j + 1]
                if _SEP_RE.fullmatch(text[e:ns] or 'x') and 'orange' not in nd:
                    continue
        for cat in THING_CATEGORIES:
            if cat in d:
                out.append((cat, s, d[cat]))
    out.sort(key=lambda t: t[1])
    return tuple(out)


def categories_in(text):
    """[(category, char_start, is_plural)] for every category mention, sorted by position.

    Multi-word terms win over their parts (longest n-gram first); null phrases such as
    "baseball cap" or "ski pole" are skipped."""
    return list(_categories_in(text))


# ----------------------------------------------------------------------------
# Head chunk
# ----------------------------------------------------------------------------
_DET_RE = re.compile(r'^(?:(?:a|an|the|this|that|these|those|some|his|her|their|its|my|your|our|another|'
                     r'each|one|any)\s+)+', re.I)
_CUT_WORDS = (
    r'with|without|holding|wearing|carrying|riding|sitting|standing|on|in|at|near|next to|beside|besides|'
    r'behind|under|underneath|above|below|that|which|who|whose|while|from|by|over|across|along|between|'
    r'inside|outside|atop|against|around|through|into|onto|upon|towards|toward|during|after|before|for|to|'
    r'among|amid|alongside|beneath|using|as|where|when|and then|'
    r'walking|running|playing(?!\s+field)|eating|looking|lying|laying|resting|parked|perched|covered|filled|'
    r'topped|sporting|hanging|stacked|placed|positioned|located|featuring|containing|adorned|decorated|'
    r'facing|leaning|flying(?![\s-]+(?:disc|disk|saucer))|swinging|jumping|crouching|performing|taking|waiting|'
    r'drinking|grazing|floating|sailing|posing|smiling|watching|reading|talking|pulling|pushing|'
    r'making|preparing|cutting|serving|displaying|showing|wielding|balancing|kneeling|squatting|'
    r'skiing|surfing|snowboarding|skateboarding|pitching|batting|catching|throwing|hitting|approaching|'
    r'crossing|passing|looking|sleeping|seated|stands|sits|holds|wears|rides|carries|has|have|is|are|'
    r'dressed|clad|situated|attached|connected|full of|'
    r'loaded|lined|surrounded|mounted|set up|set|wrapped|tied|hung|propped|lit|illuminated|'
    r'swimming(?![\s-]+(?:pool|trunks|suit))|feeding|chewing|licking|pointing|steering|operating|'
    r'engaged|dangling|poised|ready|about to|up|down|out|off|away|nearby|beyond|past|via|per|than')
_CUT_RE = re.compile(r'(?<![\w\'-])(?:%s)(?![\w-])|,|;|\s[-–—]\s|[()]' % _CUT_WORDS, re.I)
_FIRST_WORD_RE = re.compile(r'\s*[^\s,;()]+')


def head_chunk(phrase):
    p = phrase.strip()
    p = _DET_RE.sub('', p)
    m = _FIRST_WORD_RE.match(p)
    pos = m.end() if m else 0
    cut = None
    while True:
        cut = _CUT_RE.search(p, pos)
        if cut is None:
            break
        if cut.group() == ',':
            before = p[:cut.start()]
            # "large, rectangular suitcase": a short adjective run before the comma is not a head
            if len(before.split()) <= 3 and not _categories_in(before):
                nxt = _CUT_RE.match(p, cut.end()) or _CUT_RE.match(p, cut.end() + 1)
                if not nxt:
                    pos = cut.end()
                    continue
        break
    head = p[:cut.start()] if cut else p
    return head.strip(' ,;:.-')


# ----------------------------------------------------------------------------
# Status / flags / subject type
# ----------------------------------------------------------------------------
def phrase_category_status(phrase, category):
    head = head_chunk(phrase)
    cats = {c for c, _, _ in categories_in(head)}
    if category in cats:
        return 'match'
    if cats:
        return 'mismatch'
    return 'unknown'


_NUM = r'two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|many|multiple|numerous|various|both|few|dozens?'
_OF_WORDS = (r'pair|pairs|row|rows|stack|stacks|pile|piles|set|sets|bunch|bunches|cluster|clusters|line|lines|'
             r'collection|assortment|pack|array|batch|bundle|number|lot|lots|series|selection|variety|'
             r'gathering|rest|trio|duo|dozen|dozens|couple|group|groups|herd|herds|flock|flocks|team|teams|'
             r'crowd|crowds|audience|pride|school|swarm|handful|multitude|mass|bevy|gaggle|string|cluster|'
             r'bunches|heap|heaps|mound|batch')
_GROUP_RE = re.compile(
    r'(?<![\w-])(?:(?:%s)\b|(?:%s)\s+of\b|crowds?\b|herds?\b|flocks?\b|groups?\b|audience\b|couple\b|trio\b|duo\b|'
    r'[2-9]\b|\d{2,}\b)(?![\w-])' % (_NUM, _OF_WORDS), re.I)
_PAIR_EXEMPT = re.compile(r'\bpair\s+of\b(?=[^,]{0,40}?\b(?:skis?|scissors|shears|glasses|sunglasses|pants|shorts|jeans|'
                          r'trousers|binoculars|tongs|pliers|goggles)\b)', re.I)
_PLURAL_EXC = {'glass', 'dress', 'bus', 'pants', 'jeans', 'shorts', 'trousers', 'glasses', 'grass', 'class',
               'lens', 'canvas', 'chess', 'news', 'cross', 'moss', 'boss', 'mass', 'gas', 'iris', 'christmas',
               'plus', 'brass', 'kiss', 'across', 'bus', 'status', 'atlas', 'texas', 'series', 'species',
               'tennis', 'physics', 'ness', 'overalls', 'tights', 'leggings', 'pajamas', 'scissors', 'skis',
               'goggles', 'sunglasses', 'binoculars', 'stripes_', 'its', 'his', 'this', 'us', 'always',
               'perhaps', 'yes', 'thermos', 'kudos', 'gas', 'canvas', 'chassis', 'bras', 'bikini', 'swimsuits_', 'mattress', 'princess', 'actress', 'waitress',
               'hostess', 'stewardess', 'goddess', 'headdress', 'address', 'process', 'access', 'express',
               'cactus', 'bonus', 'campus', 'octopus', 'hippopotamus', 'walrus', 'lotus', 'octopus',
               'asparagus', 'citrus', 'virus', 'focus', 'circus', 'radius', 'stimulus', 'nucleus',
               'cirrus', 'cumulus', 'fungus', 'surplus', 'apparatus', 'chorus', 'genius', 'minus', 'famous'}

_PARTIAL_RE = re.compile(
    r'\bpartial(?:ly)?\b|\bpartly\b|\bpart of\b|\bedge of\b|\bcorner of\b|\bcut[\s-]*off\b|\bmostly\s+(?:hidden|obscured|'
    r'visible|cropped|out of|blocked|covered|out-of)|\bonly\b[^.]{0,30}\bvisible\b|\bobscured\b|\bpeek(?:s|ing|ed)?\b|'
    r'\bhalf[\s-]*(?:hidden|visible|cropped|out)\b|\bportion of\b|\bsection of\b|\bfragment of\b|\bsliver of\b|'
    r'\bcropped\b|\bhidden\b|\bcut\s+out\s+of\s+frame\b|\bout of frame\b|\bvisible (?:part|portion|section)\b|'
    r'\bpartial view\b|\bblocked\b|\bobstructed\b|\bbarely visible\b|\bpart\b|\bpartway\b|\bonly (?:the )?(?:top|bottom|front|back|head|legs?|tail|tip|rear|side)\b', re.I)
_PARTIAL_HEAD_RE = re.compile(
    r'^(?:the\s+)?(?:part|edge|corner|portion|section|side|back|front|top|bottom|end|tip|rear|half|'
    r'handle|legs?|hands?|arms?|head|face|wheels?|windows?|doors?|seats?|corners?|edges?|sides?|backs?|'
    r'tops?|ends?|tips?|rear end|underside|undercarriage|base|foot|feet|tail|trunk|lid|cover|body|hood)\s+of\b', re.I)
_REFLECT_RE = re.compile(r'\breflect(?:ion|ions|ed|ing|s)?\b|\bmirror(?:ed)? image\b|\bmirrored\b', re.I)
_UNCERTAIN_RE = re.compile(
    r'\bpossibly\b|\bappears?\s+to\b|\bappearing\b|\bwhat\s+(?:looks|appears|seems)\b|\blooks?\s+like\b|\blikely\b|'
    r'\bseems?\b|\bsome\s+(?:kind|sort|type)\s+of\b|\bunidentified\b|\bperhaps\b|\bprobably\b|\bmaybe\b|'
    r'\bunclear\b|\bindistinct\b|\bor something\b|\bunknown\b|\bresembl(?:es|ing)\b|\bmight be\b|\bsort of\b|\bkind of\b|\bunrecognizable\b|\bindeterminate\b', re.I)

_GENERIC_PLURAL_SKIP = re.compile(r'(?:ss|us|is|ous|ics)$')


def phrase_flags(phrase, category):
    flags = set()
    head = head_chunk(phrase)
    hl = head.lower()
    mentions = categories_in(head)
    # --- group
    group = False
    h2 = _PAIR_EXEMPT.sub('pair_of_ok', head)
    if _GROUP_RE.search(h2):
        group = True
    cat_mentions = [(s, pl) for c, s, pl in mentions if c == category]
    if category not in ('skis', 'scissors'):
        if any(pl for _, pl in cat_mentions):
            group = True
    for (s1, _), (s2, _) in zip(cat_mentions, cat_mentions[1:]):
        if re.search(r'\band\b|&|,', head[s1:s2]):
            group = True
    if not mentions and not group:
        words = re.findall(r"[a-z]+", hl)
        if words:
            last = words[-1]
            if (len(last) > 3 and last.endswith('s') and last not in _PLURAL_EXC and
                    not _GENERIC_PLURAL_SKIP.search(last) and 'of' not in words[-2:-1]):
                group = True
    if group:
        flags.add('group')
    # --- partial
    if _PARTIAL_RE.search(phrase) or _PARTIAL_HEAD_RE.search(head):
        flags.add('partial')
    if _REFLECT_RE.search(phrase):
        flags.add('reflection')
    if _UNCERTAIN_RE.search(phrase):
        flags.add('uncertain')
    return flags


_CAT_TYPE = {}
for _c in ('person',):
    _CAT_TYPE[_c] = 'human'
for _c in ('bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'):
    _CAT_TYPE[_c] = 'animal'
for _c in ('bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'):
    _CAT_TYPE[_c] = 'vehicle'
for _c in ('banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'):
    _CAT_TYPE[_c] = 'food'

_EXTRA_WORDS = {
    'animal': ('animal rabbit bunny monkey squirrel deer goat pig piglet fish lion tiger leopard cheetah '
               'wolf fox rat hamster turtle tortoise frog snake lizard crocodile alligator camel donkey mule '
               'llama alpaca kangaroo koala dolphin whale seal otter raccoon moose antelope zebra rhino '
                'rhinoceros hippo hippopotamus buffalo yak creature beast wildlife pet puppy kitten '
               'lamb calf foal cub insect butterfly bee spider dinosaur reptile mammal bird seagull parrot'),
    'vehicle': ('vehicle van tram trolley tractor bulldozer scooter helicopter limousine ambulance wagon '
                'cart carriage buggy bus taxi cab jeep rv camper trailer crane forklift cruiser sled'),
    'food': ('food meal dish dishes fries salad soup rice pasta bread meat steak fruit fruits vegetable '
             'vegetables cheese egg eggs breakfast lunch dinner dessert snack snacks cookie cookies pastry '
             'pastries burger hamburger cheeseburger noodles sushi taco burrito pie muffin bagel toast '
             'cereal sausage bacon pancake pancakes waffle waffles pudding ice cream chocolate candy '
             'cuisine entree appetizer side tomato tomatoes potato potatoes onion onions lettuce cucumber '
             'pepper peppers mushroom mushrooms lemon lime grape grapes strawberry strawberries berries '
             'cherry cherries pineapple melon watermelon corn beans chicken pork beef fish shrimp seafood '
             'sauce pizza sandwich croissant pretzel fruit slice slices wings fries nachos wrap curry stew '
             'omelet omelette meatball meatballs dumpling dumplings lasagna spaghetti'),
    'object': ('object item thing structure device equipment sign board box container bag machine tool '
               'furniture appliance toy shelf lamp light helmet hat shirt jacket glove basket tray plate '
               'tower building statue sculpture pole post fence wall door window screen display sculpture '
               'rock stone sculpture bridge tent flag banner pillow towel blanket mat rug curtain '
               'umbrella mirror picture painting frame poster'),
}
_EXTRA_TYPE = {}
for _t, _ws in _EXTRA_WORDS.items():
    for _w in _ws.split():
        _EXTRA_TYPE.setdefault(_w, _t)
_EXTRA_RE = re.compile(r"(?<![A-Za-z])(%s)(?:e?s)?(?![A-Za-z])" % '|'.join(
    sorted((re.escape(w) for w in _EXTRA_TYPE), key=lambda x: -len(x))), re.I)


def phrase_subject_type(phrase):
    head = head_chunk(phrase)
    best = None  # (pos, type)
    for c, s, _ in categories_in(head):
        t = _CAT_TYPE.get(c, 'object')
        if best is None or s >= best[0]:
            best = (s, t)
    for m in _EXTRA_RE.finditer(head):
        w = m.group(1).lower()
        t = _EXTRA_TYPE.get(w)
        if t is None:
            continue
        # extra words rank lower than category mentions at the same place
        if best is None or m.start() > best[0]:
            best = (m.start(), t)
    return best[1] if best else 'unknown'


# ----------------------------------------------------------------------------
# Scene tags
# ----------------------------------------------------------------------------
_TAG_STD = re.compile(r'<\s*\d+(?:\s*,\s*\d+)*\s*:\s*([^<>]*?)\s*>')
_TAG_XML = re.compile(r'<(\d+)>(.*?)</\1>', re.S)
_TAG_BARE = re.compile(r'<\s*\d+(?:\s*,\s*\d+)*\s*>')


def _plain(caption):
    t = _TAG_XML.sub(lambda m: m.group(2), caption)
    t = _TAG_STD.sub(lambda m: m.group(1), t)
    t = _TAG_BARE.sub('', t)
    return t


_INDOOR_CUES = (r'indoors?|interior|room|rooms|bedroom|bathroom|kitchen|ceiling|ceilings|countertops?|counters?|'
                r'cabinets?|cabinetry|cupboards?|shelf|shelves|bookshelf|bookshelves|lamps?|couch|sofa|'
                r'curtains?|carpets?|carpeted|rugs?|tiled|tiles|sink|toilet|stove|oven|refrigerator|fridge|'
                r'microwave|bathtub|shower|chandeliers?|fireplace|hallway|desk|desks|television|tv|laptop|'
                r'keyboard|restaurant|cafe|office|classroom|apartment|hardwood|floorboards?|wallpaper|'
                r'baseboard|living room|dining room|pillows?|bed|beds|bedding|nightstand|dresser|mantel|'
                r'tablecloth|doorway|inside|indoor setting|hospital|museum|library|gym|kitchenette|pantry')
_OUTDOOR_CUES = (r'outdoors?|outside|sky|skies|street|streets|field|fields|beach|beaches|grass|grassy|sidewalks?|'
                 r'ocean|sea|river|lake|mountains?|hills?|hillside|trees?|road|roads|highway|park|forest|snow|'
                 r'snowy|sand|sandy|waves?|clouds?|cloudy|sunny|sunset|sunrise|horizon|pavement|parking lot|'
                 r'skyline|pasture|farm|runway|tarmac|lawn|meadow|sunlight|blue sky|shore|coast|harbor|harbour|'
                 r'dock|pier|bridge|crosswalk|city street|urban|downtown|outdoor setting|cliff|valley|'
                 r'savanna|savannah|jungle|rural|countryside|countryside|garden|yard|backyard|playground|'
                 r'stadium|zoo|safari|trail|path|canal|pond|waterfront|intersection|traffic|cityscape')
_IND_RE = re.compile(r'(?<![A-Za-z])(%s)(?![A-Za-z])' % _INDOOR_CUES, re.I)
_OUT_RE = re.compile(r'(?<![A-Za-z])(%s)(?![A-Za-z])' % _OUTDOOR_CUES, re.I)

_SETTING_PATTERNS = {
    'kitchen': r'kitchens?|kitchenette|stoves?|stovetops?|fridge|refrigerators?|dishwasher|pantry|'
               r'microwave|cupboards?|cooktop|range hood|kitchen cabinets?',
    'bathroom': r'bathrooms?|toilets?|bathtubs?|restroom|washroom|urinals?|showers?(?! of)|bath tub|bidet|'
                r'bathroom sink|powder room',
    'bedroom': r'bedrooms?|(?<!flower )(?<!garden )beds?|nightstand|comforter|duvet|bedspread|bunk bed|'
               r'bedding|bedside',
    'living_room': r'living rooms?|couch|couches|sofas?|armchairs?|fireplace|coffee table|loveseat|recliner|'
                   r'entertainment center|family room|lounge|sitting room',
    'dining_restaurant': r'restaurants?|cafe|café|diner|dining rooms?|dining table|buffet|pubs?|coffee shop|'
                         r'cafeteria|bistro|banquet|eatery|food court|bakery|bar counter|dinner table|tavern|brewery',
    'office_workspace': r'offices?|desks?|laptops?|computers?|monitors?|workspace|cubicle|work ?stations?|classroom|'
                        r'conference room|meeting room|library|keyboard|computer screen|work area|study',
    'store_market': r'stores?|shops?|markets?|supermarket|grocery|groceries|shopping|malls?|boutique|stalls?|'
                    r'bazaar|vendors?|retail|storefront|shopkeeper|market stall|checkout|aisle|aisles',
    'street_urban': r'streets?|city|cities|urban|downtown|sidewalks?|skyscrapers?|alley|crosswalk|pedestrians?|'
                    r'plaza|town|storefronts?|lamp ?posts?|street lamp|cityscape|skyline|buildings|neighborhood|'
                    r'intersection|city street|cobblestone',
    'road_traffic': r'roads?|highway|traffic|freeway|lanes?|roadway|intersection|parking lot|asphalt|stop sign|'
                    r'traffic lights?|roadside|traffic signal|parking spaces?|parking meters?|motorway|street sign',
    'beach_water': r'beach(?:es)?|ocean|sea|shore|waves?|surf|lake|rivers?|pond|harbou?r|dock|pier|marina|'
                   r'body of water|waterfront|seaside|coast|lagoon|surfing|bay|canal|water ?way|swimming|'
                   r'sandy|seashore|sailing|waterway|stream|creek|waterfall',
    'park_field': r'parks?|fields?|lawn|meadow|gardens?|playground|picnic|grassy|grassland|open space|'
                  r'green space|courtyard|backyard|yard|lawns',
    'mountain_snow': r'mountains?|snow|snowy|snow-covered|skiing|skis?|slopes?|hills?|hillside|snowboard(?:ing|er)?|'
                     r'ski resort|glacier|cliff|summit|peaks?|valley|rocky terrain|snowfall|ski lift|alpine',
    'sports_venue': r'stadium|courts?|arena|gym|rink|bleachers|dugout|ballpark|ball field|baseball field|'
                    r'playing field|soccer field|football field|golf course|skate ?park|home plate|pitcher\'s mound|'
                    r'baseball game|tennis match|soccer game|basketball|tennis|baseball|batter|infield|outfield|'
                    r'umpire|scoreboard|fairway|racetrack|volleyball|skatepark|half[\s-]?pipe|tennis court|'
                    r'basketball court|diamond|soccer|cricket|rugby|polo',
    'airport': r'airports?|runway|tarmac|terminal|hangar|control tower|airfield|jet ?way|boarding|gate|'
               r'baggage claim|airstrip|taxiway|airplanes? on the ground',
    'railway': r'railways?|railroad|train station|railway station|train tracks?|trains?|locomotive|subway|'
               r'tram|metro|railcar|train yard|rail yard|commuter train',
    'farm_pasture': r'farms?|pastures?|barn|ranch|grazing|hay|paddock|livestock|corral|farmland|stable|stables|'
                    r'farmyard|fenced|haystack|cattle|barnyard|crop|crops|rural|countryside',
    'wild_nature': r'wildlife|wilderness|jungle|forest|savanna|savannah|safari|habitat|woods|zoo|enclosure|'
                   r'in the wild|national park|natural habitat|wild|woodland|bush|thicket|reserve|sanctuary|'
                   r'tall grass|marsh|swamp|rainforest',
    'sky_air': r'kites?|airplanes?|planes?|aircraft|jets?|helicopters?|parachute|paraglid\w*|balloons?|airborne|'
               r'mid-?air|in the air|flying|soaring|flies|takeoff|take-off|taking off|landing|in flight|hot air balloon',
}
# Settings that are too weak alone unless count>=2 (single occurrence is noise)
_WEAK_SETTINGS = {'street_urban', 'park_field', 'store_market', 'office_workspace', 'sports_venue',
                  'wild_nature', 'sky_air', 'mountain_snow', 'railway', 'beach_water', 'farm_pasture',
                  'dining_restaurant', 'living_room', 'road_traffic'}
_SETTING_RE = {k: re.compile(r'(?<![A-Za-z])(?:%s)(?![A-Za-z])' % v, re.I) for k, v in _SETTING_PATTERNS.items()}

_SETTING_IO = {'kitchen': 'indoor', 'bathroom': 'indoor', 'bedroom': 'indoor', 'living_room': 'indoor',
               'office_workspace': 'indoor', 'dining_restaurant': 'indoor', 'store_market': 'indoor',
               'beach_water': 'outdoor', 'park_field': 'outdoor', 'mountain_snow': 'outdoor',
               'street_urban': 'outdoor', 'road_traffic': 'outdoor', 'farm_pasture': 'outdoor',
               'wild_nature': 'outdoor', 'sky_air': 'outdoor', 'airport': None, 'railway': None,
               'sports_venue': None}
_SETTING_IO_WEIGHT = {'kitchen': 3, 'bathroom': 3, 'bedroom': 3, 'living_room': 2, 'office_workspace': 1,
                      'dining_restaurant': 1, 'store_market': 1, 'beach_water': 3, 'park_field': 2,
                      'mountain_snow': 3, 'street_urban': 2, 'road_traffic': 2, 'farm_pasture': 3,
                      'wild_nature': 3, 'sky_air': 1}
# Primary setting words that decide a setting by a single occurrence.
_STRONG_ONCE = {
    'kitchen': 1, 'bathroom': 1, 'bedroom': 1, 'living_room': 1, 'airport': 1, 'beach_water': 1,
    'mountain_snow': 1, 'farm_pasture': 1, 'railway': 1, 'dining_restaurant': 1, 'sports_venue': 1,
    'road_traffic': 1, 'street_urban': 1, 'store_market': 1, 'office_workspace': 2, 'park_field': 1,
    'wild_nature': 1, 'sky_air': 1,
}

_FOOD_FIRST_RE = re.compile(
    r'(?<![A-Za-z])(?:food|foods|meal|meals|dish|dishes|plate|plates|platter|breakfast|lunch|dinner|dessert|'
    r'desserts|snack|snacks|cuisine|pizza|pizzas|sandwich|sandwiches|burger|burgers|salad|soup|pastry|pastries|'
    r'cake|cakes|donut|donuts|doughnut|doughnuts|hot dog|hot dogs|broccoli|banana|bananas|fruit|fruits|'
    r'vegetables?|sushi|pasta|bowl of|cupcakes?|cookies?|bagels?|noodles|rice|steak|fries|toppings?|slice of|'
    r'carrots?|oranges?|apples?|treats?|sausages?|eggs?|pie|bread|cheese|meat|entree|appetizers?)(?![A-Za-z])', re.I)
_FOOD_CATS = {'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'}
_PLATE_RE = re.compile(r'(?<![A-Za-z])(?:plates?|bowls?|tables?|tray|platter|dish(?:es)?)(?![A-Za-z])', re.I)

_HUMAN_FORMS = sorted((f for f, d in _FORMS.items() if 'person' in d and f not in ('figure', 'figures', 'crowd', 'crowds')),
                      key=lambda f: -len(f))
_HUMAN_WORDS_RE = re.compile(r'(?<![A-Za-z])(?:%s)(?![A-Za-z])' % '|'.join(_term_regex(f) for f in _HUMAN_FORMS), re.I)

_ANIMAL_CATS = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}


def _first_sentence(t):
    m = re.search(r'(?<=[.!?])\s', t)
    return (t[:m.start()], t[m.end():]) if m else (t, '')


def scene_tags(caption):
    t = _plain(caption)
    first, rest = _first_sentence(t)
    ind_cues, out_cues = [], []
    iv = ov = 0
    for part, w in ((first, 2), (rest, 1)):
        for m in _IND_RE.finditer(part):
            iv += w
            ind_cues.append(m.group(0).lower())
        for m in _OUT_RE.finditer(part):
            ov += w
            out_cues.append(m.group(0).lower())
    settings = set()
    for lab, rx in _SETTING_RE.items():
        n = len(rx.findall(first)) * 2 + len(rx.findall(rest))
        if n >= _STRONG_ONCE[lab]:
            settings.add(lab)
    # setting-implied io votes
    for lab in settings:
        io = _SETTING_IO.get(lab)
        w = _SETTING_IO_WEIGHT.get(lab, 0)
        if io == 'indoor':
            iv += w
        elif io == 'outdoor':
            ov += w
    if iv > ov:
        io = 'indoor'
    elif ov > iv:
        io = 'outdoor'
    else:
        io = 'unknown'
    # food
    food = bool(_FOOD_FIRST_RE.search(first))
    if not food:
        fc = {c for c, _, _ in categories_in(t) if c in _FOOD_CATS}
        if len(fc) >= 2 and _PLATE_RE.search(t):
            food = True
    cats = {c for c, _, _ in categories_in(t)}
    people = bool(_HUMAN_WORDS_RE.search(t))
    animals = bool(cats & _ANIMAL_CATS) or any(
        _EXTRA_TYPE.get(m.group(1).lower()) == 'animal' for m in _EXTRA_RE.finditer(t))
    return {"io": io, "settings": sorted(settings), "food": food, "people": people, "animals": animals,
            "io_cues": {"indoor": sorted(set(ind_cues)), "outdoor": sorted(set(out_cues))}}


_STUFF_INDOOR = {'ceiling-merged': 2, 'floor-wood': 2, 'rug-merged': 2, 'cabinet-merged': 2, 'wall-tile': 2,
                 'shelf': 2, 'counter': 2, 'window-blind': 2, 'curtain': 2, 'pillow': 1, 'towel': 1,
                 'blanket': 1, 'mirror-stuff': 1, 'door-stuff': 1, 'wall-wood': 1, 'floor-other-merged': 1,
                 'light': 0}
_STUFF_OUTDOOR = {'sky-other-merged': 2, 'grass-merged': 2, 'sea': 2, 'sand': 2, 'snow': 2, 'river': 2,
                  'mountain-merged': 2, 'road': 2, 'pavement-merged': 2, 'tree-merged': 2, 'dirt-merged': 2,
                  'railroad': 2, 'playingfield': 2, 'gravel': 1, 'water-other': 1, 'bridge': 1, 'roof': 1,
                  'house': 1, 'fence-merged': 1, 'rock-merged': 1, 'building-other-merged': 1, 'tent': 1,
                  'platform': 1}


def scene_from_stuff(stuff_names):
    iv = sum(_STUFF_INDOOR.get(n, 0) for n in stuff_names)
    ov = sum(_STUFF_OUTDOOR.get(n, 0) for n in stuff_names)
    if iv > ov:
        return 'indoor'
    if ov > iv:
        return 'outdoor'
    return 'unknown'


# ----------------------------------------------------------------------------
# Caption-stated contact / occlusion between two tagged segments
# ----------------------------------------------------------------------------
# Only two consecutive tags of one sentence joined by nothing but a cue count, so the relation
# cannot belong to an untagged noun (", a white plate holds <sandwich>"). Proximity words
# (next to, near, beside, with, surrounded by) are not cues.
_CONTACT_CUES = [
    r'on', r'on top of', r'atop', r'upon', r'onto',
    r'(?:rest|rests|resting|sit|sits|sitting|seated|stand|stands|standing|lie|lies|lying|lay|lays|laying|placed|'
    r'positioned|set|perched|mounted|parked|stacked|piled|served|balanced|propped|leaning|leans|lean)'
    r' (?:on|upon|on top of|atop|in|inside|against|onto)',
    r'against', r'pressed against', r'attached to', r'hanging from', r'hangs from', r'hung on', r'hangs on',
    r'draped over', r'inside', r'in', r'within',
    r'(?:holds|holding|hold|held by|carries|carrying|carry|carried by|grips|gripping|grasping|grasps|clutching|'
    r'clutches|wearing|wears|worn by|rides|riding|ride|ridden by|supports|supporting|supported by|contains|'
    r'containing|contain|occupied by|touching|touches|hugging|hugs|cuddling|embracing|petting|pulling|pulled by|'
    r'cradling|cradles)',
    r'(?:also )?holds up', r'in contact with', r'shaking hands with', r'(?:filled|topped|covered|stacked|loaded) (?:with|by)',
    r'(?:adorned|decorated) with',
]
_OCCLUDE_CUES = [
    r'in front of', r'behind', r'(?:partially |partly )?(?:obscured|hidden|blocked|occluded|concealed) by',
    r'(?:partially |partly )?(?:obscuring|hiding|blocking|occluding)', r'overlapping',
]
_CUE_PRE = (r'(?:(?:,\s*)?(?:which|who|that)\s+)?(?:(?:is|are|was|were|being|been|also|partially|partly|firmly|gently|'
            r'neatly|securely|directly|appears to be|seems to be)\s+)*')
_CUE_DET = r'(?:\s+(?:a|an|the|its|his|her|their|some|several|multiple|two|three|four|a pair of|a set of))?'
_CUE_RE = [(kind, re.compile(r'^' + _CUE_PRE + r'(?:%s)' % '|'.join(cues) + _CUE_DET + r'$', re.I))
           for kind, cues in (('contact', _CONTACT_CUES), ('occlude', _OCCLUDE_CUES))]


def contact_pairs(tokens):
    """{frozenset({a, b}): 'contact' | 'occlude'} from parse_caption() tokens.

    Multi-id tags relate each of their ids ("<1, 2: two men> seated on <5: bench>").
    """
    tags = [(i, t) for i, t in enumerate(tokens) if t['type'] == 'tag' and t['ids']]
    rel = {}
    for (i, a), (j, b) in zip(tags, tags[1:]):
        gap = ''.join(t['text'] for t in tokens[i + 1:j])
        if re.search(r'[.;!?]', gap) or re.search(r'\bnot\b', gap, re.I):
            continue
        gap = gap.strip().rstrip(' ,')
        kind = next((k for k, rx in _CUE_RE if rx.match(gap)), None)
        if kind:
            for x in a['ids']:
                for y in b['ids']:
                    if x != y:
                        rel.setdefault(frozenset((x, y)), kind)
    return rel


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _validate():
    import json
    import os
    import random
    import sys
    from collections import Counter

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo, 'tools', 'pancap_viewer'))
    from pancap import parse_caption
    data = os.path.join(repo, 'data')
    meta = json.load(open(os.path.join(data, 'index', 'val.json')))
    cats = {c['id']: c for c in json.load(open(os.path.join(data, 'index', 'categories.json')))}
    assert [c['name'] for c in cats.values() if c['isthing']] == THING_CATEGORIES
    capdir = os.path.join(data, meta['captions'])
    rnd = random.Random(0)

    rows = []  # (phrase, category)
    scene_rows = []
    for key, s in meta['samples'].items():
        p = os.path.join(capdir, key + '.txt')
        if not os.path.exists(p):
            continue
        text = open(p).read()
        segs = {g['id']: g for g in s['segments']}
        stuff = {cats[g['category_id']]['name'] for g in s['segments'] if not cats[g['category_id']]['isthing']}
        scene_rows.append((text, stuff))
        for tok in parse_caption(text):
            if tok['type'] != 'tag' or len(tok['ids']) != 1:
                continue
            g = segs.get(tok['ids'][0])
            if not g or not cats[g['category_id']]['isthing'] or g.get('iscrowd'):
                continue
            if tok['text'].strip():
                rows.append((tok['text'], cats[g['category_id']]['name']))
    n = len(rows)
    print('single-id thing tags:', n, ' images:', len(scene_rows))
    st = Counter(phrase_category_status(p, c) for p, c in rows)
    print('status:', {k: '%.1f%%' % (100 * v / n) for k, v in st.items()})
    unk = Counter()
    unk_by_cat = Counter()
    for p, c in rows:
        if phrase_category_status(p, c) == 'unknown':
            h = head_chunk(p).lower().split()
            unk[h[-1] if h else '(empty)'] += 1
            unk_by_cat[c] += 1
    print('\ntop unknown head words:', unk.most_common(30))
    print('unknown by mask category:', unk_by_cat.most_common(12))
    mis = [(p, c) for p, c in rows if phrase_category_status(p, c) == 'mismatch']
    print('\nmismatch examples:')
    for p, c in rnd.sample(mis, min(25, len(mis))):
        print('   %-60s | %s -> %s' % (p[:60], c, sorted({x for x, _, _ in categories_in(head_chunk(p))})))
    mm = Counter((c, tuple(sorted({x for x, _, _ in categories_in(head_chunk(p))}))) for p, c in mis)
    print('top mismatch pairs:', mm.most_common(15))
    fl = {}
    for p, c in rows:
        for f in phrase_flags(p, c):
            fl.setdefault(f, []).append((p, c))
    print('\nflags:')
    for f in ('group', 'partial', 'reflection', 'uncertain'):
        v = fl.get(f, [])
        print(' %s: %d (%.2f%%)' % (f, len(v), 100 * len(v) / n))
        for p, c in rnd.sample(v, min(10, len(v))):
            print('      %-70s | %s' % (p[:70], c))
    tags = [scene_tags(t) for t, _ in scene_rows]
    N = len(tags)
    print('\nio:', {k: '%.1f%%' % (100 * v / N) for k, v in Counter(t['io'] for t in tags).items()})
    sc = Counter(s for t in tags for s in t['settings'])
    print('settings:', {k: '%.1f%%' % (100 * v / N) for k, v in sc.most_common()})
    print('settings per image:', sum(len(t['settings']) for t in tags) / N, ' none:',
          sum(1 for t in tags if not t['settings']) / N)
    for k in ('food', 'people', 'animals'):
        print(k, '%.1f%%' % (100 * sum(t[k] for t in tags) / N))
    fs = [scene_from_stuff(s) for _, s in scene_rows]
    print('stuff io:', {k: '%.1f%%' % (100 * v / N) for k, v in Counter(fs).items()})
    both = [(a, t['io']) for a, t in zip(fs, tags) if a != 'unknown' and t['io'] != 'unknown']
    agree = sum(a == b for a, b in both)
    print('agreement where both decided: %d/%d = %.1f%%' % (agree, len(both), 100 * agree / max(1, len(both))))
    dis = [(i, a, b) for i, ((a, b)) in enumerate((x, t['io']) for x, t in zip(fs, tags)) if a != 'unknown' and b != 'unknown' and a != b]
    for i, a, b in rnd.sample(dis, min(6, len(dis))):
        print('  DISAGREE stuff=%s cap=%s :: %s' % (a, b, _plain(scene_rows[i][0])[:230]))
        print('       stuff:', sorted(scene_rows[i][1]))
    print('\nsample tags:')
    for i in rnd.sample(range(N), 8):
        t = tags[i]
        print('  ', _plain(scene_rows[i][0])[:160].replace('\n', ' '))
        print('      ', {k: t[k] for k in ('io', 'settings', 'food', 'people', 'animals')})


if __name__ == '__main__':
    _validate()
