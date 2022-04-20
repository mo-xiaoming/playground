package main

type nameSetter interface {
	SetName(name string) bool
}
type nameGetter interface {
	GetName() (string, error)
}

func SetName(setter nameSetter, name string) bool {
	return setter.SetName(name)
}

func GetName(getter nameGetter) (string, error) {
	return getter.GetName()
}
