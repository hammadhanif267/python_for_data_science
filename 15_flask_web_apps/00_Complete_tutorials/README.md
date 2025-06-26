# 1.Overview
Flask is a lightweight and flexible Python web framework. It is designed to be simple and easy to use, especially for small to medium-sized applications. It follows the WSGI standard and uses the Jinja2 templating engine.

---
# 2.Installation
- Install Flask using pip:
- pip install flask

---
# 3.Basic App Structure
A minimal Flask app defines routes using decorators and runs a server using app.run(). The default project structure typically includes:
- app.py: Main application file
- templates/: HTML templates using Jinja2
- static/: Static assets like CSS, JS, and images

---
# 4.Routing
Flask uses the @app.route() decorator to define routes. Routes can be static or dynamic (with URL parameters).

# 5. Templates
HTML files are stored in the templates folder. Flask uses Jinja2 for templating, allowing dynamic content insertion using {{ variable }} syntax.

---
# 6.Static Files
Static files such as CSS and JavaScript are stored in the static directory and can be linked using url_for('static', filename='style.css').

---
# 7.Development Mode
Use debug=True in app.run() during development to enable live reloading and detailed error messages.

---
# 8.Common Extensions
Flask supports many extensions for added functionality:

- Flask-SQLAlchemy: Database ORM
- Flask-Login: User authentication
- Flask-WTF: Web forms with validation
- Flask-Migrate: Database migrations

---
# 9.Deployment Tips

- Always turn off debug mode in production.
- Use a production WSGI server like Gunicorn.
- Use environment variables for configuration and secrets.

---
# 10.Use Cases
Flask is ideal for:

- Prototyping
- Small-to-medium REST APIs
- Dashboard applications
- Educational or learning projects

---