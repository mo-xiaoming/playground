package main

/*
import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestContentContainsNodeIP(t *testing.T) {
	r := httptest.NewRequest("GET", "/content", nil)
	w := httptest.NewRecorder()
	const ip = "1.2.3.4"
	content(params{nodeID: ip})(w, r)

	b, err := ioutil.ReadAll(w.Body)
	require.NoError(t, err)
	require.Contains(t, string(b), ip)
	require.Equal(t, http.StatusOK, w.Code)
}

func TestContentOnlyAcceptGetMethod(t *testing.T) {
	methods := [...]string{
		http.MethodPost,
		http.MethodPut,
		http.MethodDelete,
		http.MethodTrace,
		http.MethodPatch,
		http.MethodOptions,
		http.MethodHead,
	}
	for _, m := range methods {
		t.Run(fmt.Sprintf("reject %s", m), func(t *testing.T) {
			r := httptest.NewRequest(m, "/content", nil)
			w := httptest.NewRecorder()
			const ip = "1.2.3.4"
			content(params{nodeID: ip})(w, r)

			require.Equal(t, http.StatusMethodNotAllowed, w.Code)
		})
	}

}
*/
