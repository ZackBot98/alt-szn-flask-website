import os
import sys

# Add your project directory to the sys.path
project_home = '/home/thealtsignal/alt-szn-flask-website'
if project_home not in sys.path:
    sys.path.append(project_home)

from app import app as application

# Add this for static files with absolute path
from werkzeug.middleware.shared_data import SharedDataMiddleware
application.wsgi_app = SharedDataMiddleware(application.wsgi_app, {
    '/static': os.path.join('/home/thealtsignal/alt-szn-flask-website', 'static')
})

if __name__ == "__main__":
    app.run()
