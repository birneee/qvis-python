import unittest

# import cysimdjson.cysimdjson
# import ujson
import simdjson

JSON_TEST_DATA = """{"glossary":{"title":"example glossary","GlossDiv":{"title":"S","GlossList":{"GlossEntry":{"ID":"SGML","SortAs":"SGML","GlossTerm":"Standard Generalized Markup Language","Acronym":"SGML","Abbrev":"ISO 8879:1986","GlossDef":{"para":"A meta-markup language, used to create markup languages such as DocBook.","GlossSeeAlso":["GML","XML"]},"GlossSee":"markup"}}}}}"""
JSON_TEST_DATA_2 = """{"glossary":{"title":"another example glossary"}}"""


class TestJsonSeq(unittest.TestCase):

    # def test_benchmark_ujson(self):
    #     for _ in range(1_000_000):
    #         json = ujson.loads(JSON_TEST_DATA)
    #         _ = json['glossary']['title']

    # def test_benchmark_cysimdjson_singleton(self):
    #     parser = cysimdjson.cysimdjson.JSONParser()
    #     for _ in range(1_000_000):
    #         json = parser.loads(JSON_TEST_DATA)
    #         _ = json['glossary']['title']

    # def test_benchmark_cysimdjson(self):
    #     for _ in range(1_000_000):
    #         parser = cysimdjson.cysimdjson.JSONParser()
    #         json = parser.loads(JSON_TEST_DATA)
    #         _ = json['glossary']['title']

    def test_benchmark_simdjson(self):
        parser = simdjson.Parser(late_reuse_check=True)
        for _ in range(1_000_000):
            json = parser.parse(JSON_TEST_DATA)
            _ = json['glossary']['title']
            #del json

    # def test_ujson_persistence(self):
    #     json1 = ujson.loads(JSON_TEST_DATA)
    #     json2 = ujson.loads(JSON_TEST_DATA_2)
    #
    #     title1 = json1['glossary']['title']
    #     title2 = json2['glossary']['title']
    #     self.assertNotEqual(title1, title2)

    # def test_cysimdjson_persistence(self):
    #     parser1 = cysimdjson.cysimdjson.JSONParser()
    #     parser2 = cysimdjson.cysimdjson.JSONParser()
    #
    #     json1 = parser1.loads(JSON_TEST_DATA)
    #     json2 = parser2.loads(JSON_TEST_DATA_2)
    #
    #     title1 = json1['glossary']['title']
    #     title2 = json2['glossary']['title']
    #     self.assertNotEqual(title1, title2)

    # def test_cysimdjson_singleton_overwrite(self):
    #     parser = cysimdjson.cysimdjson.JSONParser()
    #
    #     json1 = parser.loads(JSON_TEST_DATA)
    #     json2 = parser.loads(JSON_TEST_DATA_2)
    #
    #     title1 = json1['glossary']['title']
    #     title2 = json2['glossary']['title']
    #     self.assertEqual(title1, title2)

    def test_simdjson_expires(self):
        parser = simdjson.Parser(late_reuse_check=True)

        json1 = parser.parse(JSON_TEST_DATA)
        _ = parser.parse(JSON_TEST_DATA_2)

        self.assertRaises(RuntimeError, lambda: json1['glossary']['title'])


    def test_simdjson_persistence(self):
        parser = simdjson.Parser(late_reuse_check=True)

        json1 = parser.parse(JSON_TEST_DATA).as_dict()
        json2 = parser.parse(JSON_TEST_DATA_2)

        title1 = json1['glossary']['title']
        title2 = json2['glossary']['title']
        self.assertNotEqual(title1, title2)


if __name__ == '__main__':
    unittest.main()
