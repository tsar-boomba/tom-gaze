use std::{fmt::Debug, ops::{Mul, Sub}};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, PartialOrd)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T> Rect<T> {
    #[inline]
    pub const fn new(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    #[inline]
    pub fn from_points(pt1: Point<T>, pt2: Point<T>) -> Self
    where
        T: PartialOrd + Sub<Output = T> + Copy,
    {
        let x = partial_min(pt1.x, pt2.x);
        let y = partial_min(pt1.y, pt2.y);
        Self::new(
            x,
            y,
            partial_max(pt1.x, pt2.x) - x,
            partial_max(pt1.y, pt2.y) - y,
        )
    }

    #[inline]
	pub fn area(&self) -> T
	where
		T: Mul<Output = T> + Copy,
	{
		self.width * self.height
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, PartialOrd)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

#[cfg(feature = "opencv")]
impl<T> Into<opencv::core::Point_<T>> for Point<T> {
    fn into(self) -> opencv::core::Point_<T> {
        opencv::core::Point_ {
            x: self.x,
            y: self.y,
        }
    }
}

#[cfg(feature = "opencv")]
impl<T> Into<opencv::core::Rect_<T>> for Rect<T> {
    fn into(self) -> opencv::core::Point_<T> {
        opencv::core::Point_ {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        }
    }
}

#[inline(always)]
fn partial_min<T: PartialOrd>(a: T, b: T) -> T {
    if a <= b {
        a
    } else {
        b
    }
}

#[inline(always)]
fn partial_max<T: PartialOrd>(a: T, b: T) -> T {
    if b >= a {
        b
    } else {
        a
    }
}
