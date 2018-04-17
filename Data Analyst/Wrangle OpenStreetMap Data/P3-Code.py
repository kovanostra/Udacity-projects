#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from copy import copy as cp
from collections import defaultdict
from pymongo import MongoClient
import re
import codecs
import json
import pprint


############### DATA PARSING METHODS ###############
# The next two methods are responsible for the OSM file


def get_element(osm_file, tags=('node', 'way', 'relation')):
    # Yield element if it is the right type of tag

    # INPUT:
    # osm_file = the osm file of the desired area
    # tags = the desired tags

    # OUTPUT:
    # The contents of the desired tag
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def count_tags(filename):
    # Counts the tags inside the OSM file

    # INPUT:
    # filename = the OSM file

    # OUTPUT:
    # tags = a dictionary containing all tags and the times found in the file
        tree = ET.parse(filename)
        tags = dict()
        for childElem in tree.iter():
            child = childElem.tag
            if child not in tags:
                tags[child] = 1
            else:
                tags[child] += 1
        return tags


def key_type(element, keys):
    # Identifies the type of the key in an element

    # INPUT:
    # element = the element to be examined
    # keys = the dictionary containing the key types

    # OUTPUT:
    # keys = the dictionary containing the key types
    if element.tag == "tag":
        k = element.attrib["k"]
        if lower.search(k):
            keys["lower"] += 1
        elif lower_colon.search(k):
            keys["lower_colon"] += 1
        elif problemchars.search(k):
            keys["problemchars"] += 1
        else:
            keys["other"] += 1
        pass
    return keys


def process_keys(filename):
    # Counts the keys of the elements which are only in lowercase letters, have
    # a colon, problem characters and all the rest

    # INPUT:
    # filename = the OSM file

    # OUTPUT:
    # keys = the dictionary of keys with all the instances counted
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)
    return keys


def process_users(filename):
    # Find the number of unique users who contributed to the dataset

    # INPUT:
    # filename = the OSM file

    # OUTPUT:
    # users = the set of each username
    # len(users) = the total number of unique users
    users = set()
    for _, element in ET.iterparse(filename):
        if element.tag == "node":
            users.add(element.attrib["uid"])
    return users, len(users)


# Expected street types
expected = ["Odos", "Leoforos", "Street", "Avenue", "Boulevard", "Place",
            "Square", "Road", "Parkway", "Way"]

# Conversion of street abbreviations
mapping = {"St": "Street",
           "St.": "Street",
           "Str.": "Street",
           "Ave": "Avenue",
           "Ave.": "Avenue",
           "Rd.": "Road",
           "Leof.": "Leoforos (Avenue)",
           "Leof": "Leoforos (Avenue)",
           "Leoforos": "Leoforos (Avenue)",
           "L.": "Leoforos (Avenue)",
           "Λ.": "Leoforos (Avenue)",
           "Λ": "Leoforos (Avenue)",
           "Odos": "Odos (Street)"
           }

# The next two methods are responsible for transforming greek characters
# that exist in words into latin characters.


def get_conversion_pool():
    # Mapping of greek to latin characters
    poolGR = u"αβγδεζηικλμνοπρστυφωΑΒΓΔΕΖΗΙΚΛΜΝΟΠΡΣΤΥΦΩςάέήίϊΐόύϋΰώΆΈΉΊΪΌΎΫΏ"
    poolGL = "avgdeziiklmnoprstufoAVGDEZIIKLMNOPRSTYFOsaeiiiiouuuoAEIIIOYYO"
    special_chars = [[u'θ', 'th'], [u'ξ', 'ks'], [u'ψ', 'ps'], [u'χ', 'ch'],
                     [u'Θ', 'Th'], [u'Ξ', 'Ks'], [u'Ψ', 'Ps'], [u'Χ', 'Ch']]
    return dict(zip(poolGR, poolGL) + special_chars)


def convert(datasource):

    # INPUT:
    # datsource = the string containing a name of a street which is possibly
    # in greek characters

    # OUTPUT:
    # string converted from greek characters to latin
    pool = get_conversion_pool()
    output_line = []
    for character in datasource:
        if pool.has_key(character):
            output_line.append(pool[character])
        else:
            output_line.append(character)
    return "".join(output_line)


def audit_street_type(street_types, street_name):
    # This method checks whether a street name is expectted or not
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    # Checks whether the attribute of an element is street.
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    # This method identifies street tags inside nodes or ways. Presentation
    # purposes only

    # INPUT:
    # osmfile = the OSM file

    # OUTPUT:
    # street_types = all types of streets found in the osmfile
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


def update_name(name, mapping):
    # This method is responsible for converting the street names to their
    # expected formats, as seen in the "mapping" list

    # INPUT:
    # name = street name
    # mapping = the mapping list

    # OUTPUT:
    # fixed_name = the street name with correct street type, in latin
    # characters and capitalized the first letter of each word
    fixed_name = None
    if isinstance(name, unicode):
        name = convert(name)

    name_start = cp(name).split(" ")[0]
    len_start = len(name_start)
    name_end = cp(name).split(" ")[-1]
    len_end = len(name_end)

    if (name_start in mapping) and (name_end in mapping):
        end = len(name) - len_end
        fixed_name = mapping[name_start] +\
            name[len_start:end] +\
            mapping[name_end]
    elif (name_start in mapping):
        fixed_name = mapping[name_start] + name[len_start:]
    elif name_end in mapping:
        end = len(name) - len_end
        fixed_name = name[0:end] + mapping[name_end]
    elif (name[0:2] in mapping) and (name[1] == '.'):
        fixed_name = mapping[name[0:2]] + ' ' + name[2:]
    elif (sum(c.isdigit() for c in name) >= len(name) / 2):
        fixed_name = "Invalid Street Name"
    else:
        fixed_name = name
    return fixed_name.title()


def correct_street_name():
    # This method calls all the street names, converts them and prints them.
    # It is used here for presentation purposes.
    st_types = audit(SAMPLE_FILE)
    better_name = {}
    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name[name] = update_name(name, mapping)
            print name, "=>", better_name[name]
    return


CREATED = ["version", "changeset", "timestamp", "user", "uid"]


def shape_element(element):
    # The shape_element() converts the the contents of the "node" and "way"
    # tags of the xml file to python dictionaries accorrding to the following
    # desired format:

    '''
    {
    "id": "2406124091",
    "type: "node",
    "visible":"true",
    "created": {
              "version":"2",
              "changeset":"17206049",
              "timestamp":"2013-08-03T16:43:42Z",
              "user":"linuxUser16",
              "uid":"1219059"
            },
    "pos": [41.9757030, -87.6921867],
    "address": {
              "housenumber": "5157",
              "postcode": "60625",
              "street": "North Lincoln Ave"
            },
    "amenity": "restaurant",
    "cuisine": "mexican",
    "name": "La Cabana De Don Luis",
    "phone": "1 (773)-271-5176"
    }
    '''
    node = {}

    if element.tag == "node" or element.tag == "way":
        node["type"] = element.tag
        for attr_name, attr_value in element.attrib.items():
            if attr_name == "lat":
                if "pos" not in node:
                    node["pos"] = []
                node["pos"].insert(0, float(attr_value))
            elif attr_name == "lon":
                if "pos" not in node:
                    node["pos"] = []
                node["pos"].insert(-1, float(attr_value))
            elif attr_name in ["id", "visible", "type"]:
                node[attr_name] = convert(attr_value)
            else:
                if "created" not in node:
                    node["created"] = {}
                node["created"][attr_name] = convert(attr_value)
        for tag in element.iter("tag"):
            if problemchars.search(tag.attrib["k"]) or \
               double_colon.search(tag.attrib["k"]):
                pass
            else:
                if lower_colon.search(tag.attrib["k"]) and \
                   tag.attrib["k"].startswith("addr:"):
                    if "address" not in node:
                        node["address"] = {}
                    address_element = \
                        tag.attrib["k"][tag.attrib["k"].index(":") + 1:]
                    if address_element == "street":
                        name = update_name(tag.attrib["v"], mapping)
                        node["address"]["street"] = name
                    elif address_element == "city":
                        name = convert(tag.attrib["v"])
                        node["address"][address_element] = name.title()
                    elif address_element == "postcode":
                        code = tag.attrib["v"]
                        if re.match(r'^\d{5}$', code):
                            node["address"][address_element] = tag.attrib["v"]
                        elif re.search('[a-zA-Z]', code):
                            node["address"][address_element] =\
                                "Invalid Postal Code"
                        elif ''.join(code) == 5:
                            node["address"][address_element] = ''.join(code)
                        else:
                            node["address"][address_element] =\
                                "Invalid Postal Code"
                    else:
                        node["address"][address_element] = \
                            tag.attrib["v"].title()
                elif lower_colon.search(tag.attrib["k"]):
                    pre_colon = \
                        tag.attrib["k"][:tag.attrib["k"].index(":") + 1]
                    post_colon = \
                        tag.attrib["k"][tag.attrib["k"].index(":") + 1:]
                    if pre_colon not in node:
                        node[pre_colon[:-1]] = {}
                    node[pre_colon[:-1]][post_colon] = \
                        convert(tag.attrib["v"]).title()
                else:
                    node[tag.attrib["k"]] = convert(tag.attrib["v"]).title()
        for tag in element.iter("nd"):
            if "node_refs" not in node:
                node["node_refs"] = []
            node["node_refs"].append(convert(tag.attrib["ref"]))
        return node
    else:
        return None


def process_data(file_in, pretty=False):
    # This method takes as input the OSM file and uses the shape_element()
    # method to convert it to a python dictionary. Then it converts the python
    # dictionaries to JSON and passes it to the "data" variable, which later
    # will be imported to the database.

    # INPUT:
    # file_in = the OSM file

    # OUTPUT:
    # data = the python dictionary to input into the database
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2) + "\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data


if __name__ == "__main__":
    # Regular expressions that concern characters in strings
    lower = re.compile(r'^([a-z]|_)*$')
    lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
    double_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*:([a-z]|_)*$')
    problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

    # Regular expression concerning the street type
    street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

    # PARSE THE OSM FILE. CODE RUNS ONLY WITH THE SAMPLE FILE.
    # THE COMMENTED SECTION SHOWS HOW THE SAMPLE FILE WAS CREATED

    # OSM_FILE = "athens_greece.osm"
    SAMPLE_FILE = "athens_small_sample.osm"

    # k = 100  # Parameter: take every k-th top level element

    # with open(SAMPLE_FILE, 'wb') as output:
    #     output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    #     output.write('<osm>\n  ')

    #     # Write every kth top level element
    #     for i, element in enumerate(get_element(OSM_FILE)):
    #         if i % k == 0:
    #             output.write(ET.tostring(element, encoding='utf-8'))

    #     output.write('</osm>')

    # UNCOMMENT THE FOLLOWING FOR SEEING THE RESPECTIVE OUTPUTS

    # Counting all tage
    # count_tags(SAMPLE_FILE)

    # Counting key types
    # process_keys(SAMPLE_FILE)

    # Counting unique users
    # process_users(SAMPLE_FILE)

    # Correcting the street names
    # correct_street_name()

    # INSERT DATA IN THE DATABASE WITH THE NEXT METHOD

    def insert_data(infile, db):
        db.athens_map.insert_many(infile)

    client = MongoClient("mongodb://localhost:27017")
    db = client.athens_map

    # USE THE PROCESS MAP TO CONVERT THE XML TO PYTHON DICTIONARY IN ORDER TO
    # PUT INTO THE DATABASE
    data = process_data(SAMPLE_FILE, False)
    insert_data(data, db)

    # SAMPLE QUERIES START HERE

    # Display random data point in database
    db.athens_map.find_one()

    # Count all data points in database
    db.athens_map.find().count()

    # Find all unique instances
    usr = db.athens_map.aggregate([{'$group': {"_id": '$created.user',
                                               "count": {"$sum": 1}}},
                                   {'$sort': {"count": -1}},
                                   {"$limit": 50}])

    # UNCOMMENT TO PRINT USERS
    # pprint.pprint([{u['_id'], u['count']} for u in usr])

    # Identifies all city instances and also the data points that lack this
    # information
    city = db.athens_map.aggregate([{'$group': {"_id": '$address.city',
                                                "count": {"$sum": 1}}},
                                    {'$sort': {"count": -1}}])

    # UNCOMMENT TO PRINT CITIES
    # pprint.pprint([{c['_id'], c['count']} for c in city])

    # Identifies amenities in the city
    amenity = db.athens_map.aggregate([{"$match": {"amenity": {"$exists": 1}}},
                                       {'$group': {"_id": '$amenity',
                                                   "count": {"$sum": 1}}},
                                       {'$sort': {"count": -1}},
                                       {"$limit": 30}])

    # UNCOMMENT TO PRINT AMENITIES
    # pprint.pprint([{a['_id'], a['count']} for a in amenity])
