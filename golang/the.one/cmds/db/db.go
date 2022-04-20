package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	_ "github.com/go-sql-driver/mysql"
	"github.com/gorilla/mux"
	"html/template"
	"log"
	"net/http"
	"strconv"
)

/*
CREATE DATABASE theone;

welcome

CREATE USER 'mx'@'localhost' IDENTIFIED WITH mysql_native_password BY 'He1@@He1@@';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX, DROP, ALTER, CREATE TEMPORARY TABLES, LOCK TABLES ON theone.* TO 'mx'@'localhost';
GRANT FILE ON *.* TO 'mx'@'localhost';

CREATE USER 'mx'@'%' IDENTIFIED WITH mysql_native_password BY 'He1@@He1@@';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX, DROP, ALTER, CREATE TEMPORARY TABLES, LOCK TABLES ON theone.* TO 'mx'@'%';
GRANT FILE ON *.* TO 'mx'@'%';

FLUSH PRIVILEGES;

SELECT user,host FROM user;

CREATE TABLE `pages` (
	`id` int(11) unsigned NOT NULL AUTO_INCREMENT,
	`page_guid` varchar(256) NOT NULL DEFAULT '',
	`page_title` varchar(256) DEFAULT NULL,
	`page_content` mediumtext,
	`page_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	PRIMARY KEY (`id`),
	UNIQUE KEY `page_guid` (`page_guid`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;

INSERT INTO `pages` (`page_guid`, `page_title`, `page_content`) VALUES ('hello-world', 'Hello, World', 'I\'m so glad you found this page! It\'s been sitting patiently on the Internet for some time, just waiting for a visitor.');
INSERT INTO `pages` (`page_guid`, `page_title`, `page_content`) VALUES ('a-new-blog', 'A New Blog', 'I hope you enjoyed the last blog! Well brace yourself, because my latest blog is even <i>better</i> than the last!');
INSERT INTO `pages` (`page_guid`, `page_title`, `page_content`) VALUES ('lorem-ipsum', 'Lorem Ipsum', 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas sem tortor, lobortis in posuere sit amet, ornare non eros. Pellentesque vel lorem sed nisl dapibus fringilla. In pretium...');

CREATE TABLE `comments` (
	`id` int(11) unsigned NOT NULL AUTO_INCREMENT, `page_id` int(11) NOT NULL,
	`comment_guid` varchar(256) DEFAULT NULL, `comment_name` varchar(64) DEFAULT NULL, `comment_email` varchar(128) DEFAULT NULL, `comment_text` mediumtext,
	`comment_date` timestamp NULL DEFAULT NULL, PRIMARY KEY (`id`),
	KEY `page_id` (`page_id`)
) ENGINE=InnoDB;
*/

const (
	DBHost  = "skelix.ddns.net"
	DBPort  = ":3306"
	DBUser  = "mx"
	DBPass  = "He1@@He1@@"
	DBDbase = "theone"
	Port    = ":8080"
)

var database *sql.DB

type Page struct {
	Id         int
	Title      string
	RawContent string
	Content    template.HTML
	Date       string
	Comments   []Comment
	//Session Session
	GUID string
}

func (p Page) TruncatedText() template.HTML {
	const maxLength = 150
	if len(p.Content) > maxLength {
		return p.Content[:maxLength] + ` ...`
	}
	return p.Content
}

func main() {
	connStr := fmt.Sprintf("%s:%s@tcp(%s%s)/%s", DBUser, DBPass, DBHost, DBPort, DBDbase)
	db, err := sql.Open("mysql", connStr)
	if err != nil {
		log.Panic(err.Error())
	}
	if err := db.Ping(); err != nil {
		log.Panic(err.Error())
	}
	database = db
	log.Print("db connected")

	router := mux.NewRouter()
	router.HandleFunc("/api/pages", APIPages).Methods(http.MethodGet).Schemes("https")
	router.HandleFunc("/api/pages/{guid:[0-9a-z-]+}", APIPage).Methods(http.MethodGet).Schemes("https")
	router.HandleFunc("/api/comments", APICommentPost).Methods(http.MethodPost).Schemes("https")
	router.HandleFunc("/page/{guid:[0-9a-z-]+}", ServePage)
	router.HandleFunc("/", RedirIndex)
	router.HandleFunc("/home", ServeIndex)
	http.Handle("/", router)
	if err := http.ListenAndServeTLS(Port, "server.crt", "server.key", nil); err != nil {
		panic(err)
	}
}

type Comment struct {
	Id          int
	Name        string
	Email       string
	CommentText string
}

type JSONResponse struct {
	Fields map[string]string
}

func APICommentPost(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	name := r.FormValue("name")
	email := r.FormValue("email")
	comments := r.FormValue("comments")
	pageGUID := r.FormValue("guid")

	res, err := database.Exec("INSERT INTO comments SET page_id=(SELECT id FROM pages WHERE page_guid=?),comment_name=?,comment_email=?,comment_text=?", pageGUID, name, email, comments)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	id, err := res.LastInsertId()
	resp := JSONResponse{Fields: make(map[string]string)}
	resp.Fields["id"] = strconv.FormatInt(id, 10)
	resp.Fields["added"] = strconv.FormatBool(err == nil)
	jsonResp, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, string(jsonResp))
}

func APIPages(w http.ResponseWriter, r *http.Request) {
	pages, ok := PagesGet(w)
	if !ok {
		return
	}

	APIOutput, err := json.Marshal(pages)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, string(APIOutput))
}

func PageGet(w http.ResponseWriter, r *http.Request) (Page, bool) {
	vars := mux.Vars(r)
	pageGUID := vars["guid"]
	thisPage := Page{GUID: pageGUID}
	if err := database.QueryRow("SELECT id,page_title,page_content,page_date FROM pages WHERE page_guid=?", pageGUID).Scan(&thisPage.Id, &thisPage.Title, &thisPage.RawContent, &thisPage.Date); err != nil {
		http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
		return Page{}, false
	}
	thisPage.Content = template.HTML(thisPage.RawContent)

	comments, err := database.Query("SELECT id, comment_name as Name, comment_email, comment_text FROM comments WHERE page_id=?", thisPage.Id)
	if err != nil {
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return Page{}, false
	}
	defer func () { _ = comments.Close() }()
	for comments.Next() {
		var comment Comment
		if err := comments.Scan(&comment.Id, &comment.Name, &comment.Email, &comment.CommentText); err != nil {
			http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
			return Page{}, false
		}
		thisPage.Comments = append(thisPage.Comments, comment)
	}
	return thisPage, true
}

func APIPage(w http.ResponseWriter, r *http.Request) {
	thisPage, ok := PageGet(w, r)
	if !ok {
		return
	}

	APIOutput, err := json.Marshal(thisPage)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, string(APIOutput))
}

func ServeIndex(w http.ResponseWriter, r *http.Request) {
	pages, ok := PagesGet(w)
	if !ok {
		return
	}

	t, _ := template.ParseFiles("templates/index.html")
	_ = t.Execute(w, pages)
}

func PagesGet(w http.ResponseWriter) ([]Page, bool) {
	var Pages []Page
	conn, err := database.Query("SELECT page_title,page_content,page_date,page_guid FROM pages ORDER BY ? DESC", "page_date")
	if err != nil {
		log.Print(err.Error())
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return nil, false
	}
	defer func() { _ = conn.Close() }()

	for conn.Next() {
		thisPage := Page{}
		if err := conn.Scan(&thisPage.Title, &thisPage.RawContent, &thisPage.Date, &thisPage.GUID); err != nil {
			log.Print(err.Error())
			continue
		}
		thisPage.Content = template.HTML(thisPage.RawContent)
		Pages = append(Pages, thisPage)
	}
	return Pages, true
}

func ServePage(w http.ResponseWriter, r *http.Request) {
	thisPage, ok := PageGet(w, r)
	if !ok {
		return
	}

	t, _ := template.ParseFiles("templates/blog.html")
	_ = t.Execute(w, thisPage)
}

func RedirIndex(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "/home", http.StatusMovedPermanently)
}
