package main

/* mocking problems
func TestRedirect(t *testing.T) {
	var tests = []struct {
		name   string
		method string
		url    string
	}{
		{"get /", http.MethodGet, "/"},
		{"post /", http.MethodPost, "/"},
		{"head /", http.MethodHead, "/"},
		{"delete /", http.MethodDelete, "/"},
		{"put /", http.MethodPut, "/"},
		{"options /", http.MethodOptions, "/"},
		{"patch /", http.MethodPatch, "/"},
		{"trace /", http.MethodTrace, "/"},
		{"get /redirect", http.MethodGet, "/redirect"},
		{"post /redirect", http.MethodPost, "/redirect"},
		{"head /redirect", http.MethodHead, "/redirect"},
		{"delete /redirect", http.MethodDelete, "/redirect"},
		{"put /redirect", http.MethodPut, "/redirect"},
		{"options /redirect", http.MethodOptions, "/redirect"},
		{"patch /redirect", http.MethodPatch, "/redirect"},
		{"trace /redirect", http.MethodTrace, "/redirect"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequest(tt.method, tt.url, nil)
			w := httptest.NewRecorder()
			redirect(80, "1.2.3.4")(w, r)

			require.Equal(t, http.StatusMovedPermanently, w.Code)
			resp := w.Result()
			require.NotEmpty(t, resp.Header.Get("Location"))
		})
	}
}
*/

/* ipPools cannot be mocked so far
func wsConversation(t *testing.T, ws *websocket.Conn, sent []byte, recv []byte) {
	require.NoError(t, ws.WriteMessage(websocket.TextMessage, sent))
	typ, p, err := ws.ReadMessage()
	require.NoError(t, err)
	require.Equal(t, recv, p)
	require.Equal(t, websocket.TextMessage, typ)
}

func TestWS(t *testing.T) {
	s := httptest.NewServer(setupRouter(nil, "127.0.0.1:4448"))
	defer s.Close()

	url := strings.Replace(s.URL, "http", "ws", 1) + "/ws"

	ws, _, err := websocket.DefaultDialer.Dial(url, nil)
	require.NoError(t, err)
	defer func() {
		if err := ws.Close(); err != nil {
			log.Printf("ws close error: %v", err)
		}
	}()

	for i := 0; i < 10; i++ {
		wsConversation(t, ws, []byte("127.0.0.1"), []byte("{\"IP\":\"127.0.0.1\"}"))
		wsConversation(t, ws, []byte("1.2.3.4"), []byte("{\"IP\":\"127.0.0.1\"}"))
	}
}
*/
