package main

import "testing"

type mockDB struct {
	getNameFunc func() (string, error)
	setNameFunc func(string) bool
}

func (db *mockDB) GetName() (string, error) {
	return db.getNameFunc()
}

func (db *mockDB) SetName(name string) bool {
	return db.setNameFunc(name)
}

func TestX(t *testing.T) {
	const olive = "Olive"
	db := mockDB{
		getNameFunc: func() (string, error) {
			return olive, nil
		},
		setNameFunc: func(string) bool {
			return true
		},
	}
	if ok := SetName(&db, olive); !ok {
		t.Fail()
	}
	if name, err := GetName(&db); err != nil {
		t.Fail()
	} else if name != olive {
		t.Fail()
	}
}

func Fib(n int) int {
	if n == 1 || n == 2 {
		return 1
	}
	return Fib(n-1) + Fib(n-2)
}

func TestFib(t *testing.T) {
	var fibTests = []struct {
		n        int
		expected int
	}{
		{1, 1},
		{2, 1},
		{3, 2},
		{4, 3},
		{5, 5},
		{6, 8},
		{7, 13},
	}
	for _, tt := range fibTests {
		actual := Fib(tt.n)
		if actual != Fib(tt.n) {
			t.Errorf("Fib(%d): expected %d, got %d", tt.n, tt.expected, actual)
		}
	}
}
