package geo

import (
	"errors"
	"math"
	"net"

	"github.com/oschwald/maxminddb-golang"
)

const (
	oneDegress = math.Pi / 180
)

func toRadians(degree float64) float64 {
	return oneDegress * degree
}

type GeoLocation struct {
	Latitude  float64 `maxminddb:"latitude"`
	Longitude float64 `maxminddb:"longitude"`
}

// https://en.wikipedia.org/wiki/Haversine_formula
func Distance(pt1 GeoLocation, pt2 GeoLocation) float64 {
	var lat1 = toRadians(pt1.Latitude)
	var long1 = toRadians(pt1.Longitude)
	var lat2 = toRadians(pt2.Latitude)
	var long2 = toRadians(pt2.Longitude)

	var dlat = lat2 - lat1
	var dlong = long2 - long1

	const earthRadius = 6371

	var ans = math.Pow(math.Sin(dlat/2), 2) + math.Cos(lat1)*math.Cos(lat2)*math.Pow(math.Sin(dlong/2), 2)
	return 2 * math.Asin(math.Sqrt(ans)) * earthRadius
}

func IP2GeoLocation(ip string) (GeoLocation, error) {
	db, err := maxminddb.Open("./artifacts/GeoLite2-City.mmdb")
	if err != nil {
		return GeoLocation{}, err
	}
	defer func() {
		_ = db.Close()
	}()

	r := net.ParseIP(ip)
	if r == nil {
		return GeoLocation{}, errors.New("not valid ip")
	}

	var record struct {
		Location GeoLocation `maxminddb:"location"`
	}
	if err := db.Lookup(r, &record); err != nil {
		return GeoLocation{}, err
	}

	return record.Location, nil
}
