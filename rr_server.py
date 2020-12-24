import asyncio
import itertools

import tornado.ioloop
import tornado.web

import rr_main


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        # print(self.request.files.items())
        file_path = None
        result = "N/A"
        for filename, file in self.request.files.items():
            print(filename)
            file_path = "/home/kpatel/uploads/" + filename
            output_file = open(file_path, 'wb')
            output_file.write(file[0]['body'])
        if 'Points' in self.request.headers.keys():
            # print(self.request.headers['Points'].replace("[", "").replace("]", "").split(","))
            pts_array_str = self.request.headers['Points'].replace("[", "").replace("]", "").split(",")
            pts_array = [int(float(numeric_string)) for numeric_string in pts_array_str]
            initBB = list()
            initBB.append(pts_array[0])
            initBB.append(pts_array[1])
            initBB.append(pts_array[4] - pts_array[0])
            initBB.append(pts_array[5] - pts_array[1])
            print(initBB)
            result = rr_main.main(file_path, initBB)
        self.write(result)
        return


class TestHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello World, RR Server Works Fine!")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/test", TestHandler)
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
