use ndarray::NdFloat;

pub fn sigmoid<T: NdFloat>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}
