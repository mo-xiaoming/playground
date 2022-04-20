package geo_test

import (
	"testing"

	"github.com/mo-xiaoming/torv/internal/geo"
	"github.com/stretchr/testify/require"
)

func TestDistance(t *testing.T) {
	var pt1 = geo.GeoLocation{
		Latitude:  53.32055555555556,
		Longitude: -1.7297222222222221,
	}
	var pt2 = geo.GeoLocation{
		Latitude: 53.31861111111111,
		Longitude: -1.6997222222222223,
	}

	require.InEpsilon(t, 2.0043678382716137, geo.Distance(pt1, pt2), 0.0001)
}

func TestIP2GeoLocation(t *testing.T) {
	_, err := geo.IP2GeoLocation("127")
	require.Error(t, err)

	_, err = geo.IP2GeoLocation("127.0.0.1")
	require.NotNil(t, err)

	_, err = geo.IP2GeoLocation("18.179.111.253")
	require.NotNil(t, err)
}