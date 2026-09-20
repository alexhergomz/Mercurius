"""NOLEX: authored fact set and templates.

Everything in this file is written for this project. It contains no data from
NoLiMa (Adobe Research License, non-commercial) and no text from any other
benchmark, so it and the runner can be released under the same permissive
licence as the rest of Mercurius.

NOLEX = NO LEXical overlap, the property the needle set is built to enforce.

The design borrows NoLiMa's IDEA -- a needle whose question shares no words with
it, so the model must infer a latent association rather than match a string --
which is a benchmark design, not copyrightable expression. The needle set, the
templates, the haystack source, the scoring and the axes below are ours.

Each fact is a chain: an anchor (a landmark), the city it is in, and the country.
The question can therefore name the anchor verbatim (a literal match), the city
(one hop of world knowledge), or the country (two hops), giving a GRADED overlap
axis where NoLiMa has a binary one.
"""

# (anchor, city, country). Deliberately very well known, so that a 0.8B model
# has a chance of holding the association at all; the per-fact feasibility
# calibration in the runner drops the ones it does not.
FACTS = [
    ("the Eiffel Tower",            "Paris",            "France"),
    ("the Louvre",                  "Paris",            "France"),
    ("the Colosseum",               "Rome",             "Italy"),
    ("the Trevi Fountain",          "Rome",             "Italy"),
    ("the Leaning Tower",           "Pisa",             "Italy"),
    ("the Brandenburg Gate",        "Berlin",           "Germany"),
    ("the Semper Opera House",      "Dresden",          "Germany"),
    ("the Sagrada Familia",         "Barcelona",        "Spain"),
    ("the Prado Museum",            "Madrid",           "Spain"),
    ("the Alhambra",                "Granada",          "Spain"),
    ("Big Ben",                     "London",           "the United Kingdom"),
    ("the Acropolis",               "Athens",           "Greece"),
    ("the Kremlin",                 "Moscow",           "Russia"),
    ("the Hermitage Museum",        "Saint Petersburg", "Russia"),
    ("the Blue Mosque",             "Istanbul",         "Turkey"),
    ("the Charles Bridge",          "Prague",           "the Czech Republic"),
    ("the Anne Frank House",        "Amsterdam",        "the Netherlands"),
    ("the Atomium",                 "Brussels",         "Belgium"),
    ("the Little Mermaid statue",   "Copenhagen",       "Denmark"),
    ("the Vasa Museum",             "Stockholm",        "Sweden"),
    ("the Kiasma museum",           "Helsinki",         "Finland"),
    ("Wawel Castle",                "Krakow",           "Poland"),
    ("the Chain Bridge",            "Budapest",         "Hungary"),
    ("the Belem Tower",             "Lisbon",           "Portugal"),
    ("the Golden Gate Bridge",      "San Francisco",    "the United States"),
    ("the Statue of Liberty",       "New York City",    "the United States"),
    ("the Space Needle",            "Seattle",          "the United States"),
    ("the CN Tower",                "Toronto",          "Canada"),
    ("the Christ the Redeemer statue", "Rio de Janeiro", "Brazil"),
    ("the Sydney Opera House",      "Sydney",           "Australia"),
    ("the Burj Khalifa",            "Dubai",            "the United Arab Emirates"),
    ("the Taj Mahal",               "Agra",             "India"),
    ("the Forbidden City",          "Beijing",          "China"),
    ("the Senso-ji temple",         "Tokyo",            "Japan"),
    ("Gyeongbokgung Palace",        "Seoul",            "South Korea"),
    ("the Petronas Towers",         "Kuala Lumpur",     "Malaysia"),
    ("the Grand Palace",            "Bangkok",          "Thailand"),
    ("the Pyramids of Giza",        "Giza",             "Egypt"),
]

# Verified single-token under this tokenizer with a leading space, which is what
# makes forced-choice scoring one forward per sample: the whole candidate set is
# read off the final position's logits.
NAMES = ["Stuart", "Katie", "Veronica", "Gary", "Megan", "Calvin", "Diana",
         "Caleb", "Oscar", "Nina", "Felix", "Rosa", "Hugo", "Iris", "Victor",
         "Laura", "Simon", "Clara", "Marta", "Dennis", "Pablo", "Elena",
         "Bruno", "Sofia", "Tomas", "Alice", "Peter"]

NEEDLE = "By the way, {name} has lived right beside {anchor} for many years."

# The overlap ladder. L0 repeats the anchor verbatim and is the feasibility
# control (NoLiMa's ONLYDirect analogue); L1 and L2 share NO content word with
# the needle, so they can only be answered by inferring the association.
LEVELS = {
    "L0_verbatim": "Which person lives right beside {anchor}?",
    "L1_onehop":   "Which person lives in {city}?",
    "L2_twohop":   "Which person lives in {country}?",
}

TEMPLATE = ("Read the passage, then answer the question with a single name.\n\n"
            "{haystack}\n\nQuestion: {question}\nAnswer:")
